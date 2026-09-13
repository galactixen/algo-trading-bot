# Algo Trading Bot

An algorithmic momentum trading system built on the [Alpaca](https://alpaca.markets/) API. Includes a backtesting engine, a live paper-trading executor, a parameter grid-search optimizer, and a couple of standalone research/debug scripts.

## What's in this repo

| File | What it does |
|---|---|
| `SHORT.py` | Core backtesting engine. Simulates the VWAP / EMA(50) / MACD / ATR pullback strategy against historical data pulled from Alpaca, with sector exposure limits, regime detection, slippage modeling, and performance analytics. |
| `AAPLTEST5.py` | Live paper-trading bot. Runs the same core strategy in real time against the Alpaca paper trading API, with bracket orders, session guards, daily trade caps, and consecutive-loss limits. |
| `grid_search.py` | Parameter optimizer. Grid-searches stop-loss / take-profit / cooldown combinations per ticker and outputs the best-performing set. |
| `aapl_max_tester.py` | Standalone experiment with a mean-reversion strategy (RSI + EMA200 + Bollinger Bands) on AAPL over a 2-year window. |
| `aapl_debug_scalper.py` | Small debug/trace tool; prints a step-by-step simulated trade log for a simple VWAP + MACD scalping setup. |

## Strategy overview

The core strategy (used in `SHORT.py`, `AAPLTEST5.py`, and `grid_search.py`) looks for pullback entries in an established uptrend:

- Price above VWAP and EMA(50)
- A recent bullish MACD crossover
- A shallow pullback toward VWAP/EMA(50) within a defined band
- Trend strength confirmation (ADX / directional indicators)

Exits are managed with a hard stop-loss, a staged trailing stop once the position is in profit, and an end-of-day flatten rule.

## Setup

1. **Clone the repo**

git clone https://github.com/galactixen/algo-trading-bot.git
cd algo-trading-bot

2. **Install dependencies**

pip3 install -r requirements.txt

3. **Set up your API keys**

Create a file named `.env` in the project root (this file is gitignored and will never be committed):

ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

Get paper trading keys for free from the [Alpaca dashboard](https://app.alpaca.markets/).

4. **Run a script**

python3 SHORT.py
python3 AAPLTEST5.py
python3 grid_search.py

## Disclaimer

This project is for educational and research purposes. It trades against Alpaca's paper trading environment by default. Nothing here is financial advice, and past backtest performance is not indicative of future results.
