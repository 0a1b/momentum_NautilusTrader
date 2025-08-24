#!/usr/bin/env python3
"""
Walk-forward backtest core for Top-1 momentum strategy using NautilusTrader engine.
This module is designed to be called from a lightweight Jupyter notebook to avoid
notebook JSON corruption issues. It evaluates multiple indices and parameter sets,
plots results per index, and returns aggregated DataFrames for further analysis.
"""
from __future__ import annotations

import os
import sys
import math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Ensure local strategy module is importable
sys.path.append(os.path.abspath('stock_enhanced'))
from nautilus_engine_momentum import MomentumConfig, MomentumStrategy  # type: ignore

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig  # type: ignore
from nautilus_trader.model.identifiers import InstrumentId, Symbol  # type: ignore
from nautilus_trader.model.venues import Venue  # type: ignore
from nautilus_trader.model.instruments.equity import Equity  # type: ignore
from nautilus_trader.model.objects import Currency, Price, Quantity, Money  # type: ignore
from nautilus_trader.model.enums import CurrencyType, OmsType, AccountType, BookType  # type: ignore
from nautilus_trader.model.data import Bar, BarType  # type: ignore

# Index fetchers (Wikipedia scraping)
import importlib
idx = importlib.import_module('index_fetchers')

# ------------------------------
# Utilities
# ------------------------------

def get_adjusted_close(tickers: List[str], start: str | pd.Timestamp, end: str | pd.Timestamp, min_coverage: float = 0.6) -> pd.DataFrame:
    """Download Adjusted Close, clean, and return a wide DataFrame.
    Columns are tickers, index is business-day DatetimeIndex.
    Drops columns with coverage below `min_coverage` within [start, end].
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    # Suppress yfinance noisy prints (failed downloads etc.)
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        data = yf.download(tickers, start=start_ts, end=end_ts, progress=False, group_by='column', auto_adjust=True, threads=False)
    if isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.get_level_values(0):
        close = data['Adj Close']
    else:
        close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    # Align to business days and forward/back fill
    bidx = pd.bdate_range(start=start_ts, end=end_ts, tz=None)
    close = close.reindex(bidx).sort_index()
    # Only forward-fill to avoid inventing pre-start data; don't backfill across large gaps
    close = close.ffill()
    # Drop symbols with insufficient coverage but be lenient
    valid_counts = close.notna().sum(axis=0)
    coverage = valid_counts / len(close.index)
    keep = coverage[coverage >= min(min_coverage, 0.5)].index
    close = close[keep]
    if len(keep) == 0 and close.shape[1] > 0:
        print('  Warning: no symbols met coverage threshold; keeping top 20 by coverage')
        coverage_sorted = coverage.sort_values(ascending=False)
        keep = coverage_sorted.head(20).index
        close = close[keep]
    return close


def to_instrument_id(t: str) -> InstrumentId:
    return InstrumentId(Symbol(t), Venue('SIM'))


def to_equity(iid: InstrumentId) -> Equity:
    cur = Currency('USD', 2, 840, 'US Dollar', CurrencyType.FIAT)
    px_inc = Price(0.01, 2)
    lot = Quantity(1, 0)
    import time
    ts = int(time.time_ns())
    return Equity(iid, iid.symbol, cur, 2, px_inc, lot, ts, ts)


def bars_from_series(s: pd.Series, iid: InstrumentId):
    """Emit LAST bars only (reduces event volume by ~3x)."""
    s = s.dropna()
    last_bt = BarType.from_str(f'{iid.value}-1-DAY-LAST-EXTERNAL')
    for ts, px in s.items():
        ts_ns = pd.Timestamp(ts).tz_localize('UTC').value
        p = Price(float(px), 2)
        q = Quantity(100, 0)  # non-zero volume helps bar execution
        yield Bar(last_bt, p, p, p, p, q, ts_ns, ts_ns)


from pandas.tseries.offsets import BDay

import contextlib

def run_engine_for_close(
    close_df: pd.DataFrame,
    lookback: int,
    roc: int,
    reb_days: int,
    pos_size: float,
    trade_start: pd.Timestamp | str,
    trade_end: pd.Timestamp | str,
    initial_capital: float,
    tx_cost: float,
    quiet: bool = True,
) -> Tuple[Optional[object], Optional[MomentumStrategy]]:
    """Run Nautilus engine for given close DataFrame and parameters.
    Includes a warm-up period equal to `lookback` business days before `trade_start` to build momentum.
    The first rebalance is disallowed before `trade_start`.
    Returns (result, strategy)."""
    trade_start_ts = pd.to_datetime(trade_start)
    trade_end_ts = pd.to_datetime(trade_end)
    warmup_start = trade_start_ts - BDay(lookback)
    # Clamp warmup start to available data start and avoid dropping columns with partial NaNs
    data_start = close_df.index.min() if len(close_df.index) > 0 else trade_start_ts
    start_idx = max(warmup_start, data_start)
    c = close_df.loc[start_idx:trade_end_ts]
    c = c.dropna(axis=1, how='all')  # keep columns with some data during the period
    if c.empty or c.shape[1] == 0:
        return None, None

    def _do_run():
        cfg = BacktestEngineConfig()
        # Disable all logging completely
        import logging
        logging.getLogger('nautilus_trader').setLevel(logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        engine = BacktestEngine(cfg)

        cur = Currency('USD', 2, 840, 'US Dollar', CurrencyType.FIAT)
        engine.add_venue(
            Venue('SIM'), OmsType.NETTING, AccountType.CASH, [Money(initial_capital, cur)],
            base_currency=cur, book_type=BookType.L1_MBP, bar_execution=True,
            reject_stop_orders=True, support_gtd_orders=True, support_contingent_orders=True,
            use_position_ids=True, use_random_ids=False, use_reduce_only=False,
        )

        iids = [to_instrument_id(t) for t in c.columns]
        for iid in iids:
            engine.add_instrument(to_equity(iid))

        # Disallow first rebalance before trade_start to ensure warm-up
        min_ts_ns = int(pd.Timestamp(trade_start_ts, tz='UTC').value)
        strat_cfg = MomentumConfig(
            instrument_ids=iids,
            lookback_period=lookback,
            roc_period=roc,
            num_stocks=1,
            rebalance_days=reb_days,
            position_size=pos_size,
            transaction_cost=tx_cost,
            liquidate_on_last_bar=True,
            min_rebalance_timestamp_ns=min_ts_ns,
        )
        strategy = MomentumStrategy(strat_cfg)
        engine.add_strategy(strategy)

        bars = []
        for col, iid in zip(c.columns, iids):
            bars.extend(list(bars_from_series(c[col], iid)))
        bars.sort(key=lambda b: b.ts_event)
        engine.add_data(bars)

        import io
        import os
        # More aggressive output suppression - redirect to null device
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
            engine.run()
            result = engine.get_result()
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(null_fd)
            os.close(old_stdout)
            os.close(old_stderr)
        return result, strategy

    # Suppress noisy engine INFO logs (and setup logs) in notebooks if quiet
    if quiet:
        # Still suppress engine logs but avoid redirecting global stdout/stderr in notebooks
        return _do_run()
    else:
        return _do_run()


def extract_net_pnl(result: object) -> float:
    stats_pnls = getattr(result, 'stats_pnls', {})
    net: Optional[float] = None
    if isinstance(stats_pnls, dict) and len(stats_pnls) > 0:
        for _, v in stats_pnls.items():
            if isinstance(v, dict):
                for key in ('pnl_total', 'total', 'net', 'PnL', 'pnl'):
                    if key in v:
                        net = (net or 0.0) + float(v[key])
                        break
        if net is not None:
            return float(net)
    # Fallback: try generic stats object attributes
    for attr in ('net_pnl', 'pnl_total', 'total_pnl', 'net'):
        if hasattr(result, attr):
            try:
                return float(getattr(result, attr))
            except Exception:
                pass
    return 0.0


def compute_net_pnl_from_trade_log(trade_log: List[dict], fee: float = 0.0) -> float:
    """Compute realized PnL from a trade log with 'BUY'/'SELL' entries.
    Assumes simple inventory accounting with single-lot cost basis per symbol.
    """
    if not trade_log:
        return 0.0
    # Sort by timestamp if present
    try:
        trade_log_sorted = sorted(trade_log, key=lambda r: r.get('ts', 0))
    except Exception:
        trade_log_sorted = trade_log
    # positions: sym -> (qty, avg_cost)
    pos: Dict[str, Tuple[int, float]] = {}
    realized = 0.0
    for r in trade_log_sorted:
        side = str(r.get('side', '')).upper()
        sym = r.get('sym') or r.get('symbol')
        qty = int(r.get('qty') or r.get('quantity') or 0)
        price = float(r.get('price') or 0.0)
        if qty <= 0 or price <= 0 or not sym:
            continue
        if side == 'BUY':
            q0, c0 = pos.get(sym, (0, 0.0))
            new_qty = q0 + qty
            # Update average cost
            new_cost = ((q0 * c0) + (qty * price) + fee) / max(new_qty, 1)
            pos[sym] = (new_qty, new_cost)
        elif side == 'SELL':
            q0, c0 = pos.get(sym, (0, 0.0))
            sell_qty = min(q0, qty) if q0 > 0 else qty
            realized += (price - c0) * sell_qty - fee
            pos[sym] = (max(q0 - sell_qty, 0), c0)
    return float(realized)


def walk_forward_splits(start: str, end: str, test_days: int, step_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate successive [test_start, test_end] tuples between start and end."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while cur < end_ts:
        t_end = cur + pd.Timedelta(days=test_days)
        if t_end > end_ts:
            t_end = end_ts
        if (t_end - cur).days >= 10:  # require at least 10 days
            windows.append((cur, t_end))
        cur = cur + pd.Timedelta(days=step_days)
    return windows


from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
import pickle

def _run_engine_subprocess(args_pickle_path: str, result_pickle_path: str) -> None:
    """Run engine in subprocess to completely isolate from notebook."""
    import pickle
    with open(args_pickle_path, 'rb') as f:
        args = pickle.load(f)
    
    result = _single_param_fold(args)
    
    with open(result_pickle_path, 'wb') as f:
        pickle.dump(result, f)

def _single_param_fold(args: tuple) -> tuple[str, dict] | None:
    """Worker for parallel evaluation: runs one index + window + param set."""
    (code, close, t0, t1, p, initial_capital, tx_cost) = args
    try:
        result, strat = run_engine_for_close(
            close_df=close,
            lookback=int(p['LOOKBACK']),
            roc=int(p['ROC']),
            reb_days=int(p['REB_DAYS']),
            pos_size=float(p['POS_SIZE']),
            trade_start=t0,
            trade_end=t1,
            initial_capital=initial_capital,
            tx_cost=tx_cost,
        )
        if result is None:
            return None
        net = extract_net_pnl(result)
        # Fallback to reconstruct using strategy trade log if result stats are empty
        if (net == 0.0) and (strat is not None) and hasattr(strat, 'trade_log') and len(getattr(strat, 'trade_log')) > 0:
            try:
                net = compute_net_pnl_from_trade_log(strat.trade_log, fee=float(p.get('TX_COST', tx_cost)))
            except Exception:
                pass
        ret_pct = (net / initial_capital * 100.0) if (net is not None and initial_capital > 0) else math.nan
        row = {
            'index': code,
            'start': pd.to_datetime(t0),
            'end': pd.to_datetime(t1),
            'LOOKBACK': int(p['LOOKBACK']),
            'ROC': int(p['ROC']),
            'REB_DAYS': int(p['REB_DAYS']),
            'POS_SIZE': float(p['POS_SIZE']),
            'net_pnl': net,
            'ret_pct': ret_pct,
        }
        return (code, row)
    except Exception:
        return None


def run_walkforward(
    global_start: str,
    global_end: str,
    param_grid: List[Dict[str, float | int]],
    candidate_indices: List[str],
    test_window_days: int,
    step_days: int,
    initial_capital: float = 100_000.0,
    tx_cost: float = 7.0,
    max_workers: int | None = None,
    produce_plots: bool = False,
    verbose: bool = False,
    run_in_parallel: bool = False,  # default to sequential for engine stability in notebooks
) -> Dict[str, pd.DataFrame]:
    """Run walk-forward across indices and parameter sets.
    Returns a map of index -> aggregated DataFrame (avg net_pnl and ret_pct per param).
    Also produces bar charts per index for top parameter sets by average return%."""
    all_index_results: Dict[str, pd.DataFrame] = {}

    for code in candidate_indices:
        print(f"\nEvaluating index: {code}")
        if verbose:
            print(f"  Global range: {global_start} -> {global_end}, windows: {test_window_days}/{step_days}")
        tickers = idx.get_index_constituents(code)
        close = get_adjusted_close(tickers, global_start, global_end)
        if close.shape[1] == 0:
            print('  No valid symbols, skipping.')
            continue

        windows = walk_forward_splits(global_start, global_end, test_window_days, step_days)
        if not windows:
            print('  No windows produced, skipping.')
            continue

        rows = []
        # Build all tasks for this index
        tasks = []
        for (t0, t1) in windows:
            for p in param_grid:
                tasks.append((code, close, t0, t1, p, initial_capital, tx_cost))

        if run_in_parallel:
            if verbose:
                print(f"  Running {len(tasks)} tasks in parallel with {max_workers or 4} workers...")
            
            # Use subprocess-based parallel execution to avoid logging issues
            def run_task_subprocess(task):
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as args_file:
                    pickle.dump(task, args_file)
                    args_path = args_file.name
                
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as result_file:
                    result_path = result_file.name
                
                try:
                    # Run subprocess with timeout and complete output suppression
                    subprocess.run([
                        'python3', '-c', 
                        f'import sys; sys.path.append("{os.path.dirname(__file__)}"); '
                        f'from momentum_walkforward_core import _run_engine_subprocess; '
                        f'_run_engine_subprocess("{args_path}", "{result_path}")'
                    ], timeout=300, capture_output=True, check=True)
                    
                    with open(result_path, 'rb') as f:
                        out = pickle.load(f)
                    return out
                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    return None
                finally:
                    # Cleanup temp files
                    try:
                        os.unlink(args_path)
                        os.unlink(result_path)
                    except Exception:
                        pass
            
            with ThreadPoolExecutor(max_workers=max_workers or 4) as ex:
                for fut in as_completed(ex.submit(run_task_subprocess, t) for t in tasks):
                    try:
                        out = fut.result()
                    except Exception as e:
                        if verbose:
                            print(f"  [WARN] Task failed: {e}")
                        continue
                    if out is None:
                        continue
                    _, row = out
                    rows.append(row)
                    if verbose and len(rows) % 5 == 0:
                        print(f"    Completed {len(rows)}/{len(tasks)} tasks")
        else:
            if verbose:
                print(f"  Running {len(tasks)} tasks sequentially...")
            for i, t in enumerate(tasks):
                if verbose and i % 5 == 0:
                    print(f"    Task {i+1}/{len(tasks)}: {t[3]}-{t[4]} params={t[5]}")
                
                # Run in subprocess to avoid hanging
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as args_file:
                    pickle.dump(t, args_file)
                    args_path = args_file.name
                
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as result_file:
                    result_path = result_file.name
                
                try:
                    # Run subprocess with timeout
                    subprocess.run([
                        'python3', '-c', 
                        f'import sys; sys.path.append("{os.path.dirname(__file__)}"); '
                        f'from momentum_walkforward_core import _run_engine_subprocess; '
                        f'_run_engine_subprocess("{args_path}", "{result_path}")'
                    ], timeout=300, capture_output=True, check=True)
                    
                    with open(result_path, 'rb') as f:
                        out = pickle.load(f)
                    
                    if out is not None:
                        _, row = out
                        rows.append(row)
                        if verbose:
                            print(f"      -> PnL: {row.get('net_pnl', 0):.2f}")
                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    if verbose:
                        print(f"      -> Failed: {e}")
                    continue
                finally:
                    # Cleanup temp files
                    try:
                        os.unlink(args_path)
                        os.unlink(result_path)
                    except Exception:
                        pass

        df = pd.DataFrame(rows)
        if df.empty:
            print('  No results for this index.')
            continue

        agg = df.groupby(['LOOKBACK', 'ROC', 'REB_DAYS', 'POS_SIZE'])[['net_pnl', 'ret_pct']].mean().reset_index()
        agg = agg.sort_values(by='ret_pct', ascending=False)
        all_index_results[code] = agg

        if verbose:
            print('Top parameter sets by average return%:')
            try:
                from IPython.display import display  # type: ignore
                display(agg.head(10))
            except Exception:
                print(agg.head(10).to_string(index=False))

        if produce_plots and not agg.empty:
            # Plot bar diagram of average return% for top-10 parameter sets
            topN = agg.head(10)
            try:
                labels = [f"L{r.LOOKBACK}-R{r.ROC}-D{r.REB_DAYS}" for _, r in topN.iterrows()]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(labels, topN['ret_pct'])
                ax.set_title(f'{code} - Avg Return% by Param (top 10)')
                ax.set_ylabel('Avg Return %')
                ax.set_xticklabels(labels, rotation=45, ha='right')
                plt.show()
            except Exception:
                pass

    print('\nCompleted evaluation for indices:', list(all_index_results.keys()))
    return all_index_results


if __name__ == '__main__':
    # Example quick run (can be removed).
    PARAM_GRID = [
        { 'LOOKBACK': 150, 'ROC': 150, 'REB_DAYS': 14, 'POS_SIZE': 0.95 },
        { 'LOOKBACK': 200, 'ROC': 200, 'REB_DAYS': 14, 'POS_SIZE': 0.95 },
        { 'LOOKBACK': 250, 'ROC': 250, 'REB_DAYS': 14, 'POS_SIZE': 0.95 },
    ]
    CANDIDATE_INDICES = ['SP500', 'NASDAQ100']
    run_walkforward('2022-01-01', '2024-12-31', PARAM_GRID, CANDIDATE_INDICES, test_window_days=120, step_days=60)
