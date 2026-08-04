# Quantitative Research Pipeline

A modular, rigorous quantitative research framework for generating, testing, and validating market behavior hypotheses using NASDAQ ETF data (QQQ).

## Data Limitations

> **IMPORTANT:** This pipeline operates on OHLCV (Open, High, Low, Close, Volume) data ONLY.
>
> It does NOT have access to:
> - Level II / order book data
> - Bid-ask spread information
> - Trade-level (tick) data or Time & Sales
> - True order flow or market microstructure data
> - Dark pool activity or hidden liquidity
>
> Order flow proxy hypotheses are inferred from price-volume relationships and CANNOT capture true order flow dynamics. These signals proxy order flow behavior but cannot represent bid-ask dynamics, queue position, or hidden liquidity.
>
> All results should be interpreted with these constraints in mind.

## Architecture Overview

```
Pipeline Flow:
                                                                        
  Data Fetching     Feature Engineering     Hypothesis Generation       
  (yfinance)   -->  (OHLCV features)    -->  (103 hypotheses)           
       |                                          |                      
       v                                          v                      
  Statistical Testing    FDR Correction     Walk-Forward Validation      
  (t-tests, Sharpe) --> (Benjamini-      --> (expanding window,          
                         Hochberg)           consistency ratio)           
       |                                          |                      
       v                                          v                      
  Out-of-Sample         Regime Analysis     Transaction Costs            
  (holdout test)    --> (bull/bear/      --> (spread, slippage,           
                         sideways/crisis)    market impact)               
       |                                          |                      
       v                                          v                      
  Strategy Design       Position Sizing     Risk Controls                
  (entry/exit rules)--> (half-Kelly)    --> (drawdown breaker,           
                                             correlation adj)            
       |                                                                 
       v                                                                 
  Report Generation                                                      
  (markdown + CSV)                                                       
```

## Setup and Installation

### Prerequisites

- Python 3.11+ (recommended: pyenv for version management)
- uv (Python package manager)

### Installation

```bash
# Set Python version
pyenv local 3.11.15

# Install dependencies
uv sync
```

## Usage

### Running the Full Pipeline

```bash
uv run python -m quant_research.main
```

This will:
1. Download 10 years of QQQ data from Yahoo Finance
2. Compute technical features from OHLCV data
3. Generate 103 market behavior hypotheses
4. Test each hypothesis for statistical significance
5. Apply Benjamini-Hochberg FDR correction
6. Run walk-forward validation on survivors
7. Test on out-of-sample holdout data
8. Analyze performance across market regimes
9. Apply realistic transaction costs
10. Design entry/exit rules for final survivors
11. Compute position sizes using half-Kelly criterion
12. Apply portfolio risk controls
13. Generate a comprehensive report in `results/`

### Output

Results are saved to the `results/` directory:
- `research_report.md` - Full markdown report with all findings
- `summary.csv` - Summary table for quick reference

## Module Descriptions

| Module | Description |
|--------|-------------|
| `quant_research.data.fetcher` | Downloads and caches OHLCV data from yfinance |
| `quant_research.data.features` | Computes technical features (returns, vol, RSI, MACD, etc.) |
| `quant_research.hypotheses.generator` | Generates 103 testable market hypotheses |
| `quant_research.hypotheses.catalog` | Hypothesis data structure and categories |
| `quant_research.hypotheses.signals` | Signal computation functions |
| `quant_research.testing.statistical` | T-tests, bootstrap, permutation tests, Sharpe, etc. |
| `quant_research.testing.rejection` | FDR correction and multi-criteria rejection |
| `quant_research.validation.walk_forward` | Walk-forward validation (expanding/rolling window) |
| `quant_research.validation.out_of_sample` | Pure holdout out-of-sample testing |
| `quant_research.validation.pipeline` | Orchestrates all validation stages |
| `quant_research.robustness.regime_analysis` | Market regime identification and analysis |
| `quant_research.robustness.transaction_costs` | Realistic cost modeling (spread, slippage, impact) |
| `quant_research.strategy.entries_exits` | Entry/exit rule design and backtesting |
| `quant_research.strategy.position_sizing` | Kelly criterion, volatility targeting, equal risk |
| `quant_research.strategy.risk_controls` | Portfolio-level risk management |
| `quant_research.reporting.results` | Report generation (markdown + CSV) |
| `quant_research.main` | Main pipeline orchestrator |

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test modules
uv run pytest tests/test_strategy.py -v
uv run pytest tests/test_main.py -v
uv run pytest tests/test_data.py -v
uv run pytest tests/test_hypotheses.py -v
uv run pytest tests/test_validation.py -v
```

## Disclaimer

This software is for **research and educational purposes only**. It is NOT a trading system and should NOT be used for live trading without extensive additional validation, risk management, and regulatory compliance.

Key limitations:
- Historical performance does not guarantee future results
- OHLCV data cannot capture all market dynamics
- Transaction cost models are estimates, not guarantees
- Single-asset analysis lacks cross-sectional diversification
- No consideration of regulatory constraints or tax implications
- Backtests may contain undetected biases despite rigorous methodology

Always consult qualified financial professionals before making investment decisions.
