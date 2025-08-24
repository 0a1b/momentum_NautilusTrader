#!/usr/bin/env python3
"""
Nautilus Trader engine wiring for the momentum strategy.

This creates a Strategy subclass and backtest runner using nautilus_trader's BacktestEngine.
We use daily close bars constructed from yfinance data. For simplicity we submit market orders
on the rebalance bar close (approximated with a direct position update via orders at the bar price).

Note: The engine API is largely C-extension based; configuration is kept minimal and pragmatic.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.equity import Equity
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import BarAggregation, AggregationSource, PriceType
from nautilus_trader.cache.config import CacheConfig
from nautilus_trader.data.config import DataEngineConfig
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.execution.config import ExecEngineConfig


class MomentumConfig(StrategyConfig, frozen=True):
    instrument_ids: list[InstrumentId]
    lookback_period: int
    roc_period: int
    num_stocks: int
    rebalance_days: int  # approx bi-weekly: 14
    position_size: float
    transaction_cost: float
    liquidate_on_last_bar: bool = True
    # Optional: disallow rebalancing/trading before this timestamp (ns since epoch)
    min_rebalance_timestamp_ns: int | None = None
    tag: str = "default"


class MomentumStrategy(Strategy):
    def __init__(self, config: MomentumConfig) -> None:
        super().__init__(config)
        self.instrument_ids = config.instrument_ids
        self.lookback = config.lookback_period
        self.roc = config.roc_period
        self.n = config.num_stocks
        self.rebalance_days = config.rebalance_days
        self.pos_size = config.position_size
        self.fee = config.transaction_cost

        self.prices: dict[str, list[tuple[int, float]]] = {iid.value: [] for iid in self.instrument_ids}
        self.last_rebalance_ts: int | None = None
        self.trade_log: list[dict] = []  # record submitted orders for visibility

    def _snapshot_portfolio(self, venue) -> dict:
        cash_total = 0.0
        try:
            acct = self.portfolio.account(venue)
            cash_total = float(acct.balance_free()) if acct is not None else 0.0
        except Exception:
            cash_total = 0.0
        # Mark-to-market of positions using last price
        positions_value = 0.0
        for iid in self.instrument_ids:
            try:
                cur_net = self.portfolio.net_position(iid)
                qty = int(cur_net) if cur_net is not None else 0
            except Exception:
                qty = 0
            if qty == 0:
                continue
            price_obj = None
            try:
                price_obj = self.cache.price(iid, PriceType.LAST)
            except Exception:
                price_obj = None
            price = float(price_obj) if price_obj is not None else (float(self.prices[iid.value][-1][1]) if self.prices[iid.value] else 0.0)
            positions_value += qty * price
        return {'cash': cash_total, 'equity_mtm': cash_total + positions_value}

    def on_start(self) -> None:
        self._last_bar_ts: int | None = None
        # Subscribe to daily bars for each instrument using explicit BarType
        for iid in self.instrument_ids:
            try:
                bt = BarType.from_str(f"{iid.value}-1-DAY-LAST-EXTERNAL")
                self.subscribe_bars(bt)
            except Exception:
                # Best-effort subscription; we'll still accumulate bars via on_bar when delivered
                pass

    def on_stop(self) -> None:
        # Liquidate all positions on last bar to realize PnL (for consistency with engine totals)
        if not getattr(self.config, 'liquidate_on_last_bar', True):
            return
        venue = self.instrument_ids[0].venue if self.instrument_ids else None
        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.orders.market import MarketOrder
        for iid in self.instrument_ids:
            try:
                cur_net = self.portfolio.net_position(iid)
                qty_int = int(cur_net) if cur_net is not None else 0
            except Exception:
                qty_int = 0
            if qty_int <= 0:
                continue
            instrument = self.cache.instrument(iid)
            if instrument is None:
                continue
            qty = instrument.make_qty(qty_int)
            order: MarketOrder = self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.SELL,
                quantity=qty,
            )
            self.submit_order(order)
            # Log liquidation
            price_obj = None
            try:
                price_obj = self.cache.price(iid, PriceType.LAST)
            except Exception:
                price_obj = None
            price = float(price_obj) if price_obj is not None else (float(self.prices[iid.value][-1][1]) if self.prices[iid.value] else 0.0)
            snap = self._snapshot_portfolio(venue)
            try:
                self.trade_log.append({'ts': self._last_bar_ts or 0, 'side': 'SELL', 'sym': iid.value, 'qty': int(qty), 'price': price, 'cash': snap['cash'], 'equity_mtm': snap['equity_mtm'], 'reason': 'liquidate_on_stop'})
            except Exception:
                pass

    def on_bar(self, bar: Bar) -> None:
        # Track last bar timestamp for optional liquidation on stop
        self._last_bar_ts = bar.ts_event
        iid = bar.bar_type.instrument_id
        key = iid.value
        self.prices[key].append((bar.ts_event, float(bar.close)))
        # Rebalance check on Mondays, throttle by rebalance_days interval
        dt = pd.Timestamp(bar.ts_event, unit='ns').tz_localize('UTC')
        min_ts_ns = getattr(self.config, 'min_rebalance_timestamp_ns', None)
        if self.last_rebalance_ts is None:
            # Trigger first rebalance as soon as we have enough history AND past min_rebalance_timestamp_ns
            needed = max(self.lookback, self.roc)
            have_hist = sum(1 for h in self.prices.values() if len(h) >= needed)
            enough_hist = have_hist >= max(1, self.n)
            after_min = (min_ts_ns is None) or (bar.ts_event >= min_ts_ns)
            do_reb = bool(enough_hist and after_min)
        else:
            days = (dt - pd.Timestamp(self.last_rebalance_ts, unit='ns', tz='UTC')).days
            do_reb = days >= self.rebalance_days
        if not do_reb:
            return

        # Compute momentum for all instruments with enough history
        scores: dict[str, float] = {}
        for sym, hist in self.prices.items():
            if len(hist) < max(self.lookback, self.roc):
                continue
            series = pd.Series([p for _, p in hist], index=[pd.Timestamp(ts, unit='ns', tz='UTC') for ts, _ in hist])
            past = series.iloc[-self.roc]
            cur = series.iloc[-1]
            if past <= 0 or np.isnan(past) or np.isnan(cur):
                continue
            roc = (cur - past) / past * 100.0
            roc = max(0.0, min(1.0, float(np.ceil(roc))))
            recent = series.tail(self.lookback)
            lp = np.log(recent.replace(0, np.nan).dropna())
            if len(lp) < 20:
                continue
            x = np.arange(len(lp))
            slope = np.polyfit(x, lp, 1)[0]
            # Match the notebook: don't annualize slope in the score
            score = roc * slope
            if np.isfinite(score):
                scores[sym] = float(score)

        if not scores:
            return
        # Select top N
        selected = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.n]]
        try:
            print(f"[Rebalance {dt.date()}] Selected: {selected}")
        except Exception:
            pass

        # Target equal-weight positions across selected using current bar close
        venue = bar.bar_type.instrument_id.venue
        try:
            acct = self.portfolio.account(venue)
            bal = acct.balance_total() if acct is not None else None
            total_equity = float(bal) if bal is not None else 0.0
        except Exception:
            total_equity = 0.0
        target_value = total_equity * self.pos_size / max(len(selected), 1)

        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.orders.market import MarketOrder

        # Switch-only, all-in/all-out behavior when holding top-1
        if self.n == 1 and selected:
            top_sym = selected[0]
            # Determine current holdings
            current_positions: list[tuple[InstrumentId, int]] = []
            for iid in self.instrument_ids:
                try:
                    cur_net = self.portfolio.net_position(iid)
                    cur_qty = int(cur_net) if cur_net is not None else 0
                except Exception:
                    cur_qty = 0
                if cur_qty != 0:
                    current_positions.append((iid, cur_qty))

            # Close any non-top positions fully at first sell
            had_top = False
            for iid, cur_qty in current_positions:
                if iid.value == top_sym:
                    had_top = True
                    continue
                instrument = self.cache.instrument(iid)
                if instrument is None:
                    continue
                # Resolve current price for logging
                price_obj = None
                try:
                    price_obj = self.cache.price(iid, PriceType.LAST)
                except Exception:
                    price_obj = None
                price_used = float(price_obj) if price_obj is not None else (float(self.prices[iid.value][-1][1]) if self.prices[iid.value] else 0.0)
                qty = instrument.make_qty(abs(cur_qty))
                order: MarketOrder = self.order_factory.market(
                    instrument_id=iid,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                )
                self.submit_order(order)
                snap = self._snapshot_portfolio(venue)
                try:
                    self.trade_log.append({'ts': bar.ts_event, 'side': 'SELL', 'sym': iid.value, 'qty': int(qty), 'price': price_used, 'cash': snap['cash'], 'equity_mtm': snap['equity_mtm']})
                except Exception:
                    pass

            # If we didn't have the top position already, buy it to target allocation
            if not had_top:
                # Resolve price for top_sym
                top_iid = next((iid for iid in self.instrument_ids if iid.value == top_sym), None)
                if top_iid is not None:
                    price_obj = None
                    try:
                        price_obj = self.cache.price(top_iid, PriceType.LAST)
                    except Exception:
                        price_obj = None
                    price = float(price_obj) if price_obj is not None else (float(self.prices[top_sym][-1][1]) if self.prices[top_sym] else 0.0)
                    if price > 0:
                        instrument = self.cache.instrument(top_iid)
                        if instrument is not None:
                            target_qty = int(target_value / price)
                            if target_qty > 0:
                                qty = instrument.make_qty(target_qty)
                                order: MarketOrder = self.order_factory.market(
                                    instrument_id=top_iid,
                                    order_side=OrderSide.BUY,
                                    quantity=qty,
                                )
                                self.submit_order(order)
                                snap = self._snapshot_portfolio(venue)
                                try:
                                    self.trade_log.append({'ts': bar.ts_event, 'side': 'BUY', 'sym': top_iid.value, 'qty': int(qty), 'price': price, 'cash': snap['cash'], 'equity_mtm': snap['equity_mtm']})
                                except Exception:
                                    pass

            self.last_rebalance_ts = bar.ts_event
            return

        # Fallback: original rebalance-to-target logic for N>1
        # Collect current state, prices and desired targets
        entries = []  # list of dicts with iid, sym, price, cur_qty, target_qty, delta
        for iid in self.instrument_ids:
            sym = iid.value
            # Obtain last price using cache (falls back to last bar internally)
            price_obj = None
            try:
                price_obj = self.cache.price(iid, PriceType.LAST)
            except Exception:
                price_obj = None
            price = float(price_obj) if price_obj is not None else (float(self.prices[sym][-1][1]) if self.prices[sym] else 0.0)
            if price <= 0:
                continue

            # Target qty for selected set
            target_qty = int(target_value / price) if sym in selected else 0

            # Current net position
            try:
                cur_net = self.portfolio.net_position(iid)
                cur_qty = int(cur_net) if cur_net is not None else 0
            except Exception:
                cur_qty = 0

            delta = target_qty - cur_qty
            if delta == 0:
                continue

            entries.append({
                'iid': iid,
                'sym': sym,
                'price': price,
                'cur_qty': cur_qty,
                'target_qty': target_qty,
                'delta': delta,
            })

        if not entries:
            self.last_rebalance_ts = bar.ts_event
            return

        # Submit SELL orders first to free cash
        for e in entries:
            if e['delta'] >= 0:
                continue
            iid = e['iid']
            instrument = self.cache.instrument(iid)
            if instrument is None:
                continue
            qty = instrument.make_qty(abs(e['delta']))
            order: MarketOrder = self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.SELL,
                quantity=qty,
            )
            self.submit_order(order)
            try:
                self.trade_log.append({'ts': bar.ts_event, 'side': 'SELL', 'sym': iid.value, 'qty': int(qty)})
            except Exception:
                pass

        # Submit BUY orders
        for e in entries:
            if e['delta'] <= 0:
                continue
            iid = e['iid']
            instrument = self.cache.instrument(iid)
            if instrument is None:
                continue
            qty = instrument.make_qty(int(e['delta']))
            order: MarketOrder = self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.BUY,
                quantity=qty,
            )
            self.submit_order(order)
            try:
                self.trade_log.append({'ts': bar.ts_event, 'side': 'BUY', 'sym': iid.value, 'qty': int(qty)})
            except Exception:
                pass

        self.last_rebalance_ts = bar.ts_event

