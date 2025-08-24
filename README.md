# Stock Enhanced: Advanced Momentum Strategy Suite

This repository contains a comprehensive momentum trading strategy implementation with multiple analysis tools, backtesting engines, and walk-forward optimization capabilities.

## 🚀 Overview

The suite provides:
- **Multi-Index Analysis**: Evaluate 8+ global stock indices
- **Advanced Momentum Strategy**: ROC + trend slope-based stock selection  
- **Professional Backtesting**: NautilusTrader-powered engine with realistic execution
- **Walk-Forward Optimization**: Robust parameter testing across multiple time windows
- **Comprehensive Analytics**: Risk metrics, trade analysis, and performance visualization

## 📊 Supported Indices

**Global Coverage** (constituents fetched from Wikipedia):
- **US**: SP500, SP500_IT (S&P 500 IT sector), NASDAQ100
- **Europe**: DAX40 (Germany), FTSE100 (UK), CAC40 (France)  
- **Asia**: NIFTY50 (India), KOSPI200 (South Korea)

**Data Sources**:
- Constituents: Wikipedia (with robust scrapers and fallback logic)
- Prices: Yahoo Finance with proper ticker formatting (.DE, .L, .PA, .NS, .KS suffixes)
- US tickers automatically converted (BRK.B → BRK-B)

## 🎯 Core Strategy: Momentum Selection

**Algorithm**: 
1. **ROC (Rate of Change)**: `((current_price - past_price) / past_price) * 100`
   - Capped and normalized: `max(0, min(1, ceil(roc)))`
2. **Trend Slope**: Linear regression on log prices over lookback period
   - Annualized: `(1 + slope) ** 252`
3. **Momentum Score**: `ROC × Trend_Slope`
4. **Selection**: Top N stocks by momentum score
5. **Execution**: Equal-weight positions, periodic rebalancing

**Key Parameters**:
- `LOOKBACK`: Days for trend calculation (typical: 150-250)
- `ROC`: Days for rate-of-change calculation (typical: 150-250)  
- `REB_DAYS`: Rebalancing frequency in days (typical: 7-21)
- `POS_SIZE`: Capital utilization (typical: 0.95 = 95%)
- `NUM_STOCKS`: Number of stocks to hold (1 for top-1 strategy)

## 📁 File Structure & Usage

### Core Notebooks

#### 1. `momentum_top1_nasdaq100_nautilus_v3.ipynb`
**Purpose**: Single-run backtest with basic NautilusTrader integration

**Features**:
- Simple top-1 momentum strategy on NASDAQ-100
- Fixed parameters (lookback=250, roc=250, rebalance=14 days)
- Basic trade logging and PnL reporting
- Good for understanding core strategy mechanics

**Usage**:
```python
# Key parameters in v3
LOOKBACK_PERIOD = 250
ROC_PERIOD = 250  
REBALANCE_DAYS = 14
POSITION_SIZE = 0.95
```

**Outputs**:
- Engine performance stats (Sharpe, drawdown, total return)
- Trade log with entry/exit details
- Last 5-day momentum scores (raw + annualized)

#### 2. `momentum_top1_nasdaq100_nautilus_v5.ipynb`  
**Purpose**: Enhanced backtest with improved analytics and robustness

**Key Improvements over v3**:
- **Robust Data Handling**: Better missing data management, forward-fill only
- **Enhanced Trade Analytics**: Detailed position tracking, realized vs unrealized PnL
- **Risk Metrics**: Maximum drawdown, volatility, win/loss ratios
- **Visualization**: Equity curves, momentum score evolution, position timelines
- **Error Handling**: Graceful handling of failed downloads, missing constituents

**Additional Features**:
- Portfolio value tracking at each rebalance
- Monthly performance breakdown
- Correlation analysis between momentum scores and future returns
- Sensitivity analysis for key parameters

**Usage**:
```python
# Enhanced configuration in v5
config = {
    'lookback_period': 250,
    'roc_period': 250, 
    'rebalance_frequency': 14,
    'position_size': 0.95,
    'transaction_cost': 7.0,
    'min_position_hold_days': 1  # New in v5
}
```

#### 3. `momentum_walkforward_runner.ipynb`
**Purpose**: Systematic parameter optimization with walk-forward analysis

**Core Functionality**:
- **Multi-Parameter Testing**: Grid search across parameter combinations
- **Walk-Forward Windows**: Rolling 120-day test periods with 120-day steps
- **Multi-Index Support**: Test same strategy across different global indices
- **Parallel Processing**: Subprocess-based execution to avoid Jupyter hanging
- **Comprehensive Output**: Performance tables, rankings, statistical analysis

**Configuration Example**:
```python
# Global time window
GLOBAL_START = '2022-01-01'
GLOBAL_END = '2025-07-31'

# Walk-forward setup  
TEST_WINDOW_DAYS = 120    # Each test period length
STEP_DAYS = 120           # Step between test starts

# Parameter grid for optimization
PARAM_GRID = [
    {'LOOKBACK': 150, 'ROC': 150, 'REB_DAYS': 14, 'POS_SIZE': 0.95},
    {'LOOKBACK': 200, 'ROC': 200, 'REB_DAYS': 14, 'POS_SIZE': 0.95},
    {'LOOKBACK': 250, 'ROC': 250, 'REB_DAYS': 14, 'POS_SIZE': 0.95},
    {'LOOKBACK': 200, 'ROC': 150, 'REB_DAYS': 7, 'POS_SIZE': 0.95},
    {'LOOKBACK': 200, 'ROC': 250, 'REB_DAYS': 21, 'POS_SIZE': 0.95},
]

# Indices to evaluate
CANDIDATE_INDICES = ['SP500','SP500_IT','NASDAQ100','DAX40','FTSE100','CAC40','NIFTY50','KOSPI200']

# Execution settings
INITIAL_CAPITAL = 10_000_000  # $10M for meaningful absolute PnL
TX_COST = 7.0                 # $7 per trade
```

**Key Features**:
- **Robust Execution**: Subprocess isolation prevents Jupyter kernel hangs
- **Progress Tracking**: Verbose mode shows real-time progress
- **Error Recovery**: Failed windows are skipped, execution continues  
- **Timeout Protection**: 5-minute timeout per window prevents infinite hangs

### Core Python Modules

#### `momentum_walkforward_core.py`
**Purpose**: Walk-forward backtesting engine and utilities

**Key Functions**:

1. **`run_walkforward()`**: Main orchestration function
   ```python
   def run_walkforward(
       global_start: str,           # '2022-01-01'
       global_end: str,             # '2025-07-31'  
       param_grid: List[Dict],      # Parameter combinations to test
       candidate_indices: List[str], # ['SP500', 'NASDAQ100', ...]
       test_window_days: int,       # 120
       step_days: int,              # 120
       initial_capital: float,      # 10_000_000
       tx_cost: float,              # 7.0
       run_in_parallel: bool,       # False (recommended for stability)
       verbose: bool,               # True for progress tracking
       produce_plots: bool          # False (enable for visualizations)
   ) -> Dict[str, pd.DataFrame]
   ```

2. **`get_adjusted_close()`**: Robust price data downloading
   - Handles missing tickers gracefully
   - Forward-fill only (no back-fill to avoid look-ahead bias)
   - Coverage-based filtering (keeps symbols with >50% data)
   - Automatic business day alignment

3. **`run_engine_for_close()`**: Single backtest execution
   - NautilusTrader engine setup and configuration
   - Proper warm-up period handling (prevents look-ahead)
   - Market data bar construction from price series
   - Complete stdout/stderr isolation for Jupyter stability

4. **`extract_net_pnl()`**: PnL extraction with fallbacks
   - Primary: NautilusTrader engine stats
   - Fallback: Manual calculation from trade logs
   - Handles various engine result formats

**Output Structure**:
```python
# Returns: Dict[index_name, DataFrame]
{
    'SP500': DataFrame([
        {'LOOKBACK': 150, 'ROC': 150, 'REB_DAYS': 14, 'POS_SIZE': 0.95, 
         'net_pnl': 1100782.0, 'ret_pct': 11.007817}
    ]),
    'NASDAQ100': DataFrame([...]),
    # ... more indices
}
```

#### `nautilus_engine_momentum.py`  
**Purpose**: NautilusTrader strategy implementation

**Key Classes**:

1. **`MomentumConfig`**: Strategy configuration
   ```python
   @dataclass
   class MomentumConfig:
       instrument_ids: List[InstrumentId]      # Symbols to trade
       lookback_period: int                    # Trend calculation window
       roc_period: int                         # ROC calculation window  
       num_stocks: int                         # Number of positions (1 for top-1)
       rebalance_days: int                     # Rebalancing frequency
       position_size: float                    # Capital utilization (0.95)
       transaction_cost: float                 # Fixed cost per trade ($7)
       liquidate_on_last_bar: bool            # True (realize all PnL)
       min_rebalance_timestamp_ns: int        # Earliest rebalance time
   ```

2. **`MomentumStrategy`**: Core trading logic
   - **Data Management**: Maintains price history per instrument
   - **Momentum Calculation**: ROC × trend slope methodology
   - **Rebalancing Logic**: Monday-based with frequency throttling  
   - **Order Management**: Market orders with proper position sizing
   - **Trade Logging**: Detailed execution records for analysis

**Execution Flow**:
1. `on_start()`: Subscribe to daily bars for all instruments
2. `on_bar()`: Update price history, check rebalance conditions
3. **Rebalance Trigger**: Monday + sufficient history + min interval elapsed
4. **Score Calculation**: Momentum scores for all instruments with enough data
5. **Position Changes**: Market orders to reach target allocation
6. `on_stop()`: Liquidate all positions for final PnL calculation

#### `index_fetchers.py`
**Purpose**: Wikipedia-based constituent fetching with robust parsing

**Supported Fetchers**:
- `get_sp500()`: S&P 500 companies
- `get_sp500_it()`: S&P 500 IT sector subset  
- `get_nasdaq100()`: NASDAQ-100 components
- `get_dax40()`: German DAX 40 (.DE suffix)
- `get_ftse100()`: UK FTSE 100 (.L suffix)
- `get_cac40()`: French CAC 40 (.PA suffix)
- `get_nifty50()`: Indian NIFTY 50 (.NS suffix)  
- `get_kospi200()`: Korean KOSPI 200 (.KS suffix)

**Robust Parsing Features**:
- Multiple column name variants (`Symbol`, `Ticker`, `EPIC`, etc.)
- Unicode character cleaning (invisible characters, etc.)
- Automatic ticker formatting for Yahoo Finance compatibility
- Fallback logic when table structures change

## 📈 Output Analysis

### Walk-Forward Results Table
```
LOOKBACK  ROC  REB_DAYS  POS_SIZE     net_pnl    ret_pct
     250  250        14      0.95  1,435,183      14.35
     150  150        14      0.95  1,100,782      11.01  
     200  150         7      0.95  1,045,700      10.46
     200  200        14      0.95  1,045,552      10.46
     200  250        21      0.95     -6,937      -0.07
```

**Column Definitions**:
- **LOOKBACK**: Days for trend slope calculation
- **ROC**: Days for rate-of-change calculation  
- **REB_DAYS**: Rebalancing frequency in days
- **POS_SIZE**: Fraction of capital deployed (0.95 = 95%)
- **net_pnl**: Average net profit/loss across all test windows ($)
- **ret_pct**: Average return percentage (net_pnl / initial_capital × 100)

### Performance Interpretation

**Strong Performance Indicators**:
- `ret_pct > 10%`: Excellent momentum capture
- `ret_pct 5-10%`: Good performance, consider risk metrics
- `ret_pct 0-5%`: Marginal, evaluate against benchmark

**Parameter Insights**:
- **Longer lookbacks (250)** often outperform shorter ones
- **Frequent rebalancing (7-14 days)** typically beats monthly (21+ days)
- **High position sizing (0.95)** maximizes returns but increases risk
- **Matching LOOKBACK and ROC** periods often perform well

### Risk Considerations

**Strategy Limitations**:
- **Momentum-dependent**: Performs poorly in mean-reverting markets
- **Transaction costs**: High turnover can erode profits with higher costs
- **Concentration risk**: Top-1 strategy has significant single-stock exposure
- **Market regime sensitivity**: Bull markets favor momentum, bear markets may not

**Recommended Analysis**:
1. **Drawdown analysis**: Maximum loss periods and recovery times
2. **Sharpe ratio**: Risk-adjusted returns comparison
3. **Win/loss ratios**: Frequency and magnitude of profitable periods
4. **Correlation with market**: Strategy performance vs benchmark indices

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone <repository_url>
cd stock_enhanced

# Create Python environment  
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_nautilus.txt

# Verify installation
python -c "import nautilus_trader; print('NautilusTrader ready')"
```

## 🚦 Quick Start Guide

### 1. Single Backtest (Beginner)
```bash
# Open basic momentum strategy
jupyter notebook momentum_top1_nasdaq100_nautilus_v3.ipynb

# Run all cells - takes ~2-3 minutes
# Output: Single backtest results for NASDAQ-100
```

### 2. Enhanced Analysis (Intermediate)  
```bash
# Open enhanced version with analytics
jupyter notebook momentum_top1_nasdaq100_nautilus_v5.ipynb

# Modify parameters in config cell if desired
# Run all cells - takes ~3-5 minutes  
# Output: Detailed performance metrics and visualizations
```

### 3. Parameter Optimization (Advanced)
```bash
# Open walk-forward optimizer
jupyter notebook momentum_walkforward_runner.ipynb

# Adjust PARAM_GRID and CANDIDATE_INDICES as needed
# Set run_in_parallel=False for first run
# Execute - takes 10-60 minutes depending on scope
# Output: Comprehensive parameter rankings across indices
```

## 🔧 Configuration Tips

**For Faster Testing**:
```python
# Reduce scope for quick validation
GLOBAL_END = '2023-12-31'                    # Shorter time window
CANDIDATE_INDICES = ['NASDAQ100']             # Single index
TEST_WINDOW_DAYS = 60                        # Shorter test windows
PARAM_GRID = [{'LOOKBACK': 200, 'ROC': 200, 'REB_DAYS': 14, 'POS_SIZE': 0.95}]  # Single param set
```

**For Production Analysis**:
```python
# Full scope for comprehensive results  
GLOBAL_END = '2025-07-31'                    # Full recent history
CANDIDATE_INDICES = ['SP500','NASDAQ100','DAX40','FTSE100','CAC40','NIFTY50']  # Multiple indices
run_in_parallel = True                       # Enable after testing
max_workers = 4                              # Adjust based on CPU cores
```

**Troubleshooting**:
- **Jupyter hangs**: Keep `run_in_parallel=False` and `verbose=True`
- **Memory issues**: Reduce `CANDIDATE_INDICES` or shorten time windows
- **Network errors**: Add delays between yfinance calls, check internet connection
- **Empty results**: Verify ticker symbols are valid and have sufficient price history

This comprehensive momentum strategy suite provides enterprise-grade backtesting capabilities with robust error handling, making it suitable for both research and potential production deployment.
