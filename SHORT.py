import os
import time
from datetime import datetime, timedelta, date
import pandas as pd
import pandas_ta as ta
import pytz
import numpy as np
from collections import defaultdict
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BT_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.timezone("US/Eastern"))
BT_END_DATE   = datetime(2024, 12, 31, tzinfo=pytz.timezone("US/Eastern"))
START_BALANCE = 10_000.0
RISK_PCT         = 0.01
MAX_POSITION_PCT = 1.0
SL_MULT          = 2.0
TP_MULT          = 2.41
MACD_LOOKBACK_BARS = 12
PULLBACK_BAND      = 0.004
TOP_N_TICKERS      = 5
MIN_TREND_SCORE    = 30
SLIPPAGE_BPS = 0.0005
SPY_ADX_THRESHOLD = 20
MAX_SECTOR_EXPOSURE = 2
UNIVERSE = {
    # Mega-cap Tech
    "AAPL": "TECH",  "MSFT": "TECH",  "GOOGL": "TECH",
    "META": "TECH",  "AMZN": "TECH",  "NFLX":  "TECH",
    "CRM":  "TECH",  "NOW":  "TECH",  "ADBE":  "TECH",
    "ORCL": "TECH",  "INTU": "TECH",  "PANW":  "TECH",
    "CRWD": "TECH",  "SNOW": "TECH",
    # Semiconductors
    "NVDA": "SEMI",  "AMD":  "SEMI",  "AVGO": "SEMI",
    "MU":   "SEMI",  "QCOM": "SEMI",  "ARM":  "SEMI",
    "AMAT": "SEMI",  "LRCX": "SEMI",  "KLAC": "SEMI",
    "TXN":  "SEMI",
    # Healthcare
    "LLY":  "HEALTH", "UNH":  "HEALTH", "JNJ":  "HEALTH",
    "ABBV": "HEALTH", "MRK":  "HEALTH", "AMGN": "HEALTH",
    "ISRG": "HEALTH", "NVO":  "HEALTH",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    "SLB": "ENERGY", "OXY": "ENERGY", "MPC": "ENERGY",
    # Financials
    "JPM": "FIN", "BAC": "FIN", "GS":   "FIN",
    "MS":  "FIN", "V":   "FIN", "MA":   "FIN",
    "AXP": "FIN", "SPGI":"FIN",
    # Consumer
    "TSLA": "CONS", "HD":   "CONS", "NKE":  "CONS",
    "SBUX": "CONS", "MCD":  "CONS", "COST": "CONS",
    "TGT":  "CONS",
    # Industrials
    "CAT": "INDUS", "HON": "INDUS", "UPS": "INDUS",
    "LMT": "INDUS", "GE":  "INDUS", "DE":  "INDUS",
    # Communications
    "DIS":   "COMM", "CMCSA": "COMM", "TMUS": "COMM",
}
ALL_TICKERS = list(UNIVERSE.keys())
EASTERN     = pytz.timezone("US/Eastern")
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
# ─────────────────────────────────────────────────────────────────────────────
# SLIPPAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_slippage_buy(price: float) -> float:
    return price * (1.0 + SLIPPAGE_BPS)
def apply_slippage_sell(price: float) -> float:
    return price * (1.0 - SLIPPAGE_BPS)
# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_bars(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    timeframe: TimeFrame,
    retries: int = 3,
) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
            )
            bars = data_client.get_stock_bars(req).df
            if bars.empty:
                return pd.DataFrame()
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(ticker, level="symbol")
            bars.index = pd.to_datetime(bars.index)
            bars.index = bars.index.tz_convert(EASTERN)
            bars.columns = [c.capitalize() for c in bars.columns]
            return bars
        except Exception as e:
            print(f"    [Attempt {attempt+1}] Error fetching {ticker}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = df.index.date
    vwap_series = (
        df.groupby("Date", group_keys=False)
          .apply(lambda g: ta.vwap(g["High"], g["Low"], g["Close"], g["Volume"]))
    )
    df["VWAP"] = vwap_series.values
    return df.drop(columns="Date")
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = calculate_daily_vwap(df)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    macd = ta.macd(df["Close"])
    if macd is not None and not macd.empty:
        df["MACD"]   = macd["MACD_12_26_9"]
        df["MACD_S"] = macd["MACDs_12_26_9"]
    else:
        df["MACD"] = df["MACD_S"] = 0.0
    df["ATR"]      = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None and not adx_df.empty:
        df["ADX"]      = adx_df["ADX_14"]
        df["DI_plus"]  = adx_df["DMP_14"]
        df["DI_minus"] = adx_df["DMN_14"]
    else:
        df["ADX"] = df["DI_plus"] = df["DI_minus"] = 0.0
    return df.dropna()
def build_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["EMA20"]  = ta.ema(df["Close"], length=20)
    df["EMA50"]  = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None and not adx_df.empty:
        df["ADX"]      = adx_df["ADX_14"]
        df["DI_plus"]  = adx_df["DMP_14"]
        df["DI_minus"] = adx_df["DMN_14"]
    else:
        df["ADX"] = df["DI_plus"] = df["DI_minus"] = 0.0
    df["ATR"]        = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Returns"]    = df["Close"].pct_change()
    df["Mom60"]      = df["Close"].pct_change(60)
    df["Vol60"]      = df["Returns"].rolling(60).std()
    df["RiskAdjMom"] = df["Mom60"] / (df["Vol60"] + 1e-9)
    return df.dropna()
# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_trend_quality(
    df_5min: pd.DataFrame,
    daily_df: pd.DataFrame,
    spy_daily: pd.DataFrame,
) -> float:
    recent = df_5min.tail(200).copy()
    if len(recent) < 50:
        return 0.0
    adx_score      = float(recent["ADX"].mean())
    di_score       = float((recent["DI_plus"] - recent["DI_minus"]).mean())
    above_vwap_pct = float((recent["Close"] > recent["VWAP"]).mean()) * 100
    ema_start      = float(recent["EMA50"].iloc[0])
    ema_end        = float(recent["EMA50"].iloc[-1])
    ema_slope      = ((ema_end - ema_start) / ema_start) * 10_000
    rs_score = 0.0
    if (not daily_df.empty
            and not spy_daily.empty
            and "RiskAdjMom" in daily_df.columns
            and "RiskAdjMom" in spy_daily.columns
            and len(daily_df) > 0
            and len(spy_daily) > 0):
        try:
            stock_ram = float(daily_df["RiskAdjMom"].iloc[-1])
            spy_ram   = float(spy_daily["RiskAdjMom"].iloc[-1])
            if not np.isnan(stock_ram) and not np.isnan(spy_ram):
                rs_score = max(-50.0, min(50.0, (stock_ram - spy_ram) * 100))
        except Exception:
            rs_score = 0.0
    return round(
        adx_score        * 0.40
        + di_score       * 0.20
        + above_vwap_pct * 0.15
        + ema_slope      * 0.10
        + rs_score       * 0.15,
        2
    )
# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, spy_daily: pd.DataFrame, qqq_daily: pd.DataFrame):
        self.spy    = spy_daily.copy()
        self.qqq    = qqq_daily.copy()
        self._cache: dict = {}
    def get_regime(self, today: date) -> str:
        if today in self._cache:
            return self._cache[today]
        spy_row = self._get_row(self.spy, today)
        qqq_row = self._get_row(self.qqq, today)
        if spy_row is None or qqq_row is None:
            self._cache[today] = "UNKNOWN"
            return "UNKNOWN"
        spy_above_ema50  = float(spy_row["Close"]) > float(spy_row["EMA50"])
        spy_above_ema200 = float(spy_row["Close"]) > float(spy_row["EMA200"])
        spy_adx_strong   = float(spy_row["ADX"])   > SPY_ADX_THRESHOLD
        qqq_above_ema200 = float(qqq_row["Close"]) > float(qqq_row["EMA200"])
        if spy_above_ema50 and spy_above_ema200 and spy_adx_strong and qqq_above_ema200:
            regime = "TRENDING_BULL"
        elif not spy_above_ema50:
            regime = "TRENDING_BEAR"
        else:
            regime = "CHOPPY"
        self._cache[today] = regime
        return regime
    @staticmethod
    def _get_row(df: pd.DataFrame, today: date):
        if df.empty:
            return None
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            today_ts = pd.Timestamp(today, tz=idx.tz)
        else:
            today_ts = pd.Timestamp(today)
        past = df[idx.normalize() <= today_ts]
        return past.iloc[-1] if not past.empty else None
# ─────────────────────────────────────────────────────────────────────────────
#PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_analytics(trade_log: list, start_balance: float) -> dict:
    if not trade_log:
        return {}
    pnls   = [t["P&L"] for t in trade_log]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_trades  = len(pnls)
    win_rate      = len(wins) / total_trades if total_trades else 0
    avg_win       = float(np.mean(wins))   if wins   else 0.0
    avg_loss      = float(np.mean(losses)) if losses else 0.0
    expectancy    = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    balances = [start_balance] + [t["Balance"] for t in trade_log]
    peak     = start_balance
    max_dd   = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd
    daily_pnl: dict = defaultdict(float)
    for t in trade_log:
        day = t["Exit Time"][:10]
        daily_pnl[day] += t["P&L"]
    if len(daily_pnl) > 1:
        daily_returns = np.array(list(daily_pnl.values())) / start_balance
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0.0
    return {
        "total_trades":  total_trades,
        "win_rate":      win_rate * 100,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "expectancy":    expectancy,
        "profit_factor": profit_factor,
        "max_drawdown":  max_dd * 100,
        "sharpe":        sharpe,
        "gross_profit":  gross_profit,
        "gross_loss":    gross_loss,
    }
# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest():
    print("=" * 70)
    print(f"RUNNING DYNAMIC MOMENTUM BACKTEST ({BT_START_DATE.date()} to {BT_END_DATE.date()})")
    print(f"  Universe : {len(ALL_TICKERS)} stocks across {len(set(UNIVERSE.values()))} sectors")
    print(f"  Regime   : SPY/QQQ filter ACTIVE — only trades in TRENDING_BULL")
    print(f"  Slippage : {SLIPPAGE_BPS*10000:.1f} bps per fill")
    print(f"  Sector   : Max {MAX_SECTOR_EXPOSURE} simultaneous positions per sector")
    print(f"  Exits    : Partial 50% at 1R then trail remainder")
    print(f"  Simulating Live Environment: Recalculating Top {TOP_N_TICKERS} stocks every morning.")
    print("=" * 70)
    fetch_start = BT_START_DATE - timedelta(days=120)
    data_dict    = {}   # 5-min bars
    daily_data   = {}   # daily bars with indicators
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: DOWNLOAD ALL DATA
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/2] Bulk Downloading Data for {len(ALL_TICKERS)} tickers + SPY + QQQ...")
    print("      (This may take 3–8 minutes for the expanded universe)\n")
    # Fetch SPY and QQQ daily data for regime detection
    print("  Fetching SPY (regime)...")
    spy_raw = fetch_bars("SPY", fetch_start, BT_END_DATE, TimeFrame.Day)
    spy_daily = build_daily_indicators(spy_raw) if not spy_raw.empty else pd.DataFrame()
    print("  Fetching QQQ (regime)...")
    qqq_raw = fetch_bars("QQQ", fetch_start, BT_END_DATE, TimeFrame.Day)
    qqq_daily = build_daily_indicators(qqq_raw) if not qqq_raw.empty else pd.DataFrame()

    # Build regime detector
    regime_detector = RegimeDetector(spy_daily, qqq_daily)

    # Fetch all universe tickers
    for ticker in ALL_TICKERS:
        print(f"  Fetching {ticker}...")

        # 5-minute bars
        df_5min = fetch_bars(ticker, fetch_start, BT_END_DATE, TimeFrame(5, TimeFrameUnit.Minute))
        data_dict[ticker] = build_indicators(df_5min) if not df_5min.empty else pd.DataFrame()

        # Daily bars
        df_day = fetch_bars(ticker, fetch_start, BT_END_DATE, TimeFrame.Day)
        daily_data[ticker] = build_daily_indicators(df_day) if not df_day.empty else pd.DataFrame()

    print("\n[2/2] Starting Chronological Simulation...\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: BUILD MASTER TIMESTAMP LIST
    # ─────────────────────────────────────────────────────────────────────────
    all_timestamps = set()
    for df in data_dict.values():
        if not df.empty:
            all_timestamps.update(df.index)
    all_timestamps = sorted([
        dt for dt in all_timestamps
        if dt >= BT_START_DATE
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: BACKTEST STATE
    # ─────────────────────────────────────────────────────────────────────────
    balance          = START_BALANCE
    open_positions   = {}   
    trade_log        = []
    current_date     = None
    selected_symbols = []
    bars_since_cross = {t: None for t in ALL_TICKERS}

    regime_trade_counts = defaultdict(int)
    regime_trade_pnl    = defaultdict(float)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def get_daily_row_before(ticker: str, today: date):
        """Return the most recent daily row strictly before today."""
        ddf = daily_data.get(ticker, pd.DataFrame())
        if ddf.empty:
            return None
        idx = ddf.index
        if hasattr(idx, "tz") and idx.tz is not None:
            today_ts = pd.Timestamp(today, tz=idx.tz)
        else:
            today_ts = pd.Timestamp(today)
        past = ddf[idx.normalize() < today_ts]
        return past.iloc[-1] if not past.empty else None

    def is_prior_day_bullish(ticker: str, today: date) -> bool:
        row = get_daily_row_before(ticker, today)
        if row is None:
            return False
        return float(row["Close"]) > float(row["EMA20"])

    def get_sector_counts(positions: dict) -> dict:
        """Count how many open positions exist per sector."""
        counts = defaultdict(int)
        for t in positions:
            counts[UNIVERSE.get(t, "UNKNOWN")] += 1
        return counts

    def get_spy_daily_row_before(today: date):
        """Fetch SPY daily row before today for scoring context."""
        if spy_daily.empty:
            return pd.DataFrame()
        idx = spy_daily.index
        if hasattr(idx, "tz") and idx.tz is not None:
            today_ts = pd.Timestamp(today, tz=idx.tz)
        else:
            today_ts = pd.Timestamp(today)
        past = spy_daily[idx.normalize() < today_ts]
        return past if not past.empty else pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: MAIN CHRONOLOGICAL LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for current_dt in all_timestamps:

        is_eod = (current_dt.hour >= 15 and current_dt.minute >= 30)
        is_active_hours = (
            (current_dt.hour == 10 and current_dt.minute >= 15)
            or (10 < current_dt.hour < 15)
            or (current_dt.hour == 15 and current_dt.minute < 30)
        )

        today_regime = regime_detector.get_regime(current_dt.date())

        # ─────────────────────────────────────────────────────────────────────
        # DAILY ROTATION: recalculate Top N every morning
        # ─────────────────────────────────────────────────────────────────────
        if current_date != current_dt.date():
            current_date = current_dt.date()

            # Only bother scoring if regime is tradeable
            if today_regime == "TRENDING_BULL":
                spy_slice = get_spy_daily_row_before(current_date)

                scores = {}
                for ticker in ALL_TICKERS:
                    df5 = data_dict.get(ticker, pd.DataFrame())
                    ddf = daily_data.get(ticker, pd.DataFrame())
                    if df5.empty:
                        continue

                    past_5min = df5.loc[
                        (df5.index < pd.Timestamp(current_date, tz=EASTERN))
                        & (df5.index >= pd.Timestamp(current_date - timedelta(days=60), tz=EASTERN))
                    ]

                    past_daily = pd.DataFrame()
                    if not ddf.empty:
                        idx = ddf.index
                        if hasattr(idx, "tz") and idx.tz is not None:
                            cut = pd.Timestamp(current_date, tz=idx.tz)
                        else:
                            cut = pd.Timestamp(current_date)
                        past_daily = ddf[idx.normalize() < cut]

                    scores[ticker] = score_trend_quality(past_5min, past_daily, spy_slice)

                all_scores     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                new_symbols    = [t for t, s in all_scores if s >= MIN_TREND_SCORE][:TOP_N_TICKERS]
            else:
                new_symbols = []

            for t in ALL_TICKERS:
                if t not in new_symbols:
                    bars_since_cross[t] = None

            selected_symbols = new_symbols

        # ─────────────────────────────────────────────────────────────────────
        # MANAGE OPEN POSITIONS
        # ─────────────────────────────────────────────────────────────────────
        for ticker in list(open_positions.keys()):
            df5 = data_dict.get(ticker, pd.DataFrame())
            if df5.empty or current_dt not in df5.index:
                continue

            curr_bar = df5.loc[current_dt]
            pos      = open_positions[ticker]

            high  = float(curr_bar["High"])
            low   = float(curr_bar["Low"])
            close = float(curr_bar["Close"])

            exit_reason = None
            exit_price  = 0.0

            # ── Update trailing high ──
            if high > pos["trail_high"]:
                pos["trail_high"] = high

            # ── Activate trailing stop once price moves 1×ATR above entry ──
            if not pos["trail_active"] and pos["trail_high"] >= pos["entry_price"] + pos["trail_atr"]:
                pos["trail_active"] = True

            if pos["trail_active"]:
                new_trail_sl = pos["trail_high"] - pos["trail_atr"]
                if new_trail_sl > pos["sl"]:
                    pos["sl"] = new_trail_sl

            days_held  = (current_dt.date() - pos["entry_date"]).days
            is_max_hold = days_held >= 7

            if (not pos["partial_done"]
                    and high >= pos["entry_price"] + pos["trail_atr"]):
                half_shares = pos["shares"] // 2
                if half_shares > 0:
                    partial_price  = apply_slippage_sell(pos["entry_price"] + pos["trail_atr"])
                    partial_pnl    = (partial_price - pos["entry_price"]) * half_shares
                    balance       += (pos["entry_price"] * half_shares) + partial_pnl
                    pos["shares"] -= half_shares
                    pos["partial_done"] = True

                    trade_log.append({
                        "Ticker":     ticker,
                        "Entry Time": pos["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit Time":  current_dt.strftime("%Y-%m-%d %H:%M"),
                        "Entry $":    pos["entry_price"],
                        "Exit $":     partial_price,
                        "Stop Loss $":pos["sl"],
                        "Target $":   pos["tp"],
                        "Shares":     half_shares,
                        "P&L":        partial_pnl,
                        "Result":     "✅ Partial 1R",
                        "Balance":    balance,
                        "Regime":     pos["entry_regime"],
                        "Sector":     UNIVERSE.get(ticker, "?"),
                        "ADX_entry":  pos["adx_at_entry"],
                    })

                    regime_trade_counts[pos["entry_regime"]] += 1
                    regime_trade_pnl[pos["entry_regime"]]    += partial_pnl

            # ── Determine full exit ──
            if is_eod and (is_max_hold or not pos["trail_active"]):
                exit_reason = "⏳ Max Hold Exit" if is_max_hold else "🕐 EOD Exit"
                exit_price  = apply_slippage_sell(close)
            elif low <= pos["sl"]:
                exit_reason = "✅ Trail Stop" if pos["trail_active"] else "❌ Stop Loss"
                exit_price  = apply_slippage_sell(pos["sl"])
            elif high >= pos["tp"]:
                exit_reason = "✅ Take Profit"
                exit_price  = apply_slippage_sell(pos["tp"])

            if exit_reason and pos["shares"] > 0:
                pnl      = (exit_price - pos["entry_price"]) * pos["shares"]
                balance += (pos["entry_price"] * pos["shares"]) + pnl

                trade_log.append({
                    "Ticker":     ticker,
                    "Entry Time": pos["entry_time"].strftime("%Y-%m-%d %H:%M"),
                    "Exit Time":  current_dt.strftime("%Y-%m-%d %H:%M"),
                    "Entry $":    pos["entry_price"],
                    "Exit $":     exit_price,
                    "Stop Loss $":pos["sl"],
                    "Target $":   pos["tp"],
                    "Shares":     pos["shares"],
                    "P&L":        pnl,
                    "Result":     exit_reason,
                    "Balance":    balance,
                    "Regime":     pos["entry_regime"],
                    "Sector":     UNIVERSE.get(ticker, "?"),
                    "ADX_entry":  pos["adx_at_entry"],
                })

                regime_trade_counts[pos["entry_regime"]] += 1
                regime_trade_pnl[pos["entry_regime"]]    += pnl
                del open_positions[ticker]

        # Skip new entries outside active hours or EOD
        if is_eod or not is_active_hours:
            continue
        if current_dt.hour > 14 or (current_dt.hour == 14 and current_dt.minute >= 30):
            continue

        if today_regime != "TRENDING_BULL":
            continue

        # ─────────────────────────────────────────────────────────────────────
        # SCAN FOR NEW ENTRIES
        # ─────────────────────────────────────────────────────────────────────
        sector_counts = get_sector_counts(open_positions)

        for ticker in selected_symbols:
            if ticker in open_positions:
                continue

            ticker_sector = UNIVERSE.get(ticker, "UNKNOWN")
            if sector_counts.get(ticker_sector, 0) >= MAX_SECTOR_EXPOSURE:
                continue

            df5 = data_dict.get(ticker, pd.DataFrame())
            if df5.empty or current_dt not in df5.index:
                continue

            curr_idx = df5.index.get_loc(current_dt)
            if curr_idx < 2:
                continue

            curr_bar      = df5.iloc[curr_idx]
            prev_bar      = df5.iloc[curr_idx - 1]
            prev_prev_bar = df5.iloc[curr_idx - 2]

            # ── MACD crossover detection ──
            macd_up   = (
                prev_prev_bar["MACD"] <= prev_prev_bar["MACD_S"]
                and prev_bar["MACD"]  >  prev_bar["MACD_S"]
            )
            macd_down = (
                prev_prev_bar["MACD"] >= prev_prev_bar["MACD_S"]
                and prev_bar["MACD"]  <  prev_bar["MACD_S"]
            )

            if macd_up:
                bars_since_cross[ticker] = 0
            elif macd_down:
                bars_since_cross[ticker] = None
            elif bars_since_cross[ticker] is not None:
                bars_since_cross[ticker] += 1
                if bars_since_cross[ticker] > MACD_LOOKBACK_BARS:
                    bars_since_cross[ticker] = None

            if bars_since_cross[ticker] is None or bars_since_cross[ticker] < 1:
                continue

            # ── Standard filters ──
            if float(prev_bar["ADX"])    < 25:
                continue
            if float(prev_bar["Volume"]) < float(prev_bar["Vol_MA20"]) * 1.2:
                continue
            if not is_prior_day_bullish(ticker, current_dt.date()):
                continue

            near_vwap = (
                0 <= (float(prev_bar["Close"]) - float(prev_bar["VWAP"]))
                / float(prev_bar["VWAP"]) < PULLBACK_BAND
            )
            near_ema = (
                0 <= (float(prev_bar["Close"]) - float(prev_bar["EMA50"]))
                / float(prev_bar["EMA50"]) < PULLBACK_BAND
            )

            above_vwap  = float(prev_bar["Close"]) > float(prev_bar["VWAP"])
            above_ema   = float(prev_bar["Close"]) > float(prev_bar["EMA50"])
            di_bullish  = float(prev_bar["DI_plus"]) > float(prev_bar["DI_minus"])

            if not ((near_vwap or near_ema) and above_vwap and above_ema and di_bullish):
                continue

            confirmation = float(curr_bar["Close"]) > float(prev_bar["High"])
            if not confirmation:
                continue

            if current_dt.hour > 13 or (current_dt.hour == 13 and current_dt.minute >= 30):
                continue

            entry_price = apply_slippage_buy(float(curr_bar["Open"]))

            # ── Intraday extension guard (>2% from open) ──
            today_bars = df5[df5.index.date == current_dt.date()]
            if not today_bars.empty:
                todays_open    = float(today_bars.iloc[0]["Open"])
                intraday_move  = (entry_price - todays_open) / todays_open
                if intraday_move > 0.02:
                    continue

            # ── ATR-based stops ──
            atr     = float(prev_bar["ATR"])
            atr_pct = atr / entry_price
            capped_atr = min(max(atr_pct, 0.005), 0.03) * entry_price

            sl_dist = capped_atr * SL_MULT
            tp_dist = capped_atr * TP_MULT

            ideal_shares       = int((balance * RISK_PCT) / sl_dist)
            max_allowed_shares = int((balance * MAX_POSITION_PCT) / entry_price)
            shares             = min(ideal_shares, max_allowed_shares, int(balance // entry_price))

            if shares <= 0:
                continue

            open_positions[ticker] = {
                "entry_time":    current_dt,
                "entry_date":    current_dt.date(),
                "entry_price":   entry_price,
                "sl":            entry_price - sl_dist,
                "tp":            entry_price + tp_dist,
                "shares":        shares,
                "trail_high":    entry_price,
                "trail_active":  False,
                "trail_atr":     capped_atr,
                "partial_done":  False,       
                "entry_regime":  today_regime, 
                "adx_at_entry":  float(prev_bar["ADX"]),  
            }
            balance -= entry_price * shares

            # Update sector counts immediately so next ticker sees it
            sector_counts[ticker_sector] = sector_counts.get(ticker_sector, 0) + 1
            bars_since_cross[ticker] = None

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: RESULTS — EXACT ORIGINAL FORMAT + NEW ANALYTICS BLOCK
    # ─────────────────────────────────────────────────────────────────────────

    # ── Compute analytics ─
    analytics = compute_analytics(trade_log, START_BALANCE)

    # ── Leaderboard (original format) ──
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD (FULL YEAR)")
    print("=" * 70)
    print("Ticker    Trades  Win Rate (%)  Net Profit ($)")

    stats              = []
    grand_total_pnl    = 0.0
    total_trades_all   = len(trade_log)
    total_wins_all     = len([t for t in trade_log if t["P&L"] > 0])

    for ticker in ALL_TICKERS:
        t_trades = [t for t in trade_log if t["Ticker"] == ticker]
        if not t_trades:
            continue
        wins       = [t for t in t_trades if t["P&L"] > 0]
        net_profit = sum(t["P&L"] for t in t_trades)
        grand_total_pnl += net_profit
        win_rate   = (len(wins) / len(t_trades)) * 100
        stats.append((ticker, len(t_trades), win_rate, net_profit))

    stats = sorted(stats, key=lambda x: x[3], reverse=True)

    for t, trades, wr, pnl in stats:
        print(f"  {t:>4}        {trades:>2}         {wr:>5.2f}         {pnl:>6.2f}")

    print("-" * 70)
    overall_win_rate = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0.0
    print(f"  GRAND TOTAL   {total_trades_all:>2}         {overall_win_rate:>5.2f}         {grand_total_pnl:>6.2f}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE ANALYTICS BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    if analytics:
        print("\n" + "=" * 70)
        print("PERFORMANCE ANALYTICS")
        print("=" * 70)
        print(f"  Total Trades    : {analytics['total_trades']}")
        print(f"  Win Rate        : {analytics['win_rate']:.2f}%")
        print(f"  Avg Win         : ${analytics['avg_win']:.2f}")
        print(f"  Avg Loss        : ${analytics['avg_loss']:.2f}")
        print(f"  Expectancy      : ${analytics['expectancy']:.2f}  (E = P_w*W - P_l*L)")
        print(f"  Profit Factor   : {analytics['profit_factor']:.2f}")
        print(f"  Max Drawdown    : {analytics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio    : {analytics['sharpe']:.2f}")
        print(f"  Gross Profit    : ${analytics['gross_profit']:.2f}")
        print(f"  Gross Loss      : ${analytics['gross_loss']:.2f}")
        print(f"  Final Balance   : ${balance:.2f}")
        print(f"  Net Return      : {((balance - START_BALANCE) / START_BALANCE) * 100:.2f}%")
        print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME ATTRIBUTION BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REGIME ATTRIBUTION")
    print("=" * 70)
    print(f"  {'Regime':<20} {'Trades':>6}  {'Net PnL ($)':>12}")
    print("-" * 70)
    all_regimes = set(list(regime_trade_counts.keys()))
    for reg in sorted(all_regimes):
        print(f"  {reg:<20} {regime_trade_counts[reg]:>6}  {regime_trade_pnl[reg]:>12.2f}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTOR ATTRIBUTION BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTOR ATTRIBUTION")
    print("=" * 70)
    print(f"  {'Sector':<12} {'Trades':>6}  {'Wins':>5}  {'Win Rate':>9}  {'Net PnL ($)':>12}")
    print("-" * 70)

    sector_stats: dict = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for tr in trade_log:
        sec = tr.get("Sector", "?")
        sector_stats[sec]["trades"] += 1
        sector_stats[sec]["pnl"]    += tr["P&L"]
        if tr["P&L"] > 0:
            sector_stats[sec]["wins"] += 1

    for sec, ss in sorted(sector_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        sec_wr = (ss["wins"] / ss["trades"] * 100) if ss["trades"] > 0 else 0.0
        print(f"  {sec:<12} {ss['trades']:>6}  {ss['wins']:>5}  {sec_wr:>8.2f}%  {ss['pnl']:>12.2f}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # DETAILED TRADE LOG — EXACT ORIGINAL FORMAT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED TRADE LOG")
    print("=" * 70)

    for t, total_trades, wr, pnl in stats:
        t_trades = [tr for tr in trade_log if tr["Ticker"] == t]
        wins     = len([tr for tr in t_trades if tr["P&L"] > 0])
        losses   = total_trades - wins

        print(f"\n── {t}  |  {total_trades} trades  |  {wins}W / {losses}L  |  Net: ${pnl:.2f}")
        print(f"  {'Entry Time':<16}  {'Exit Time':<16}  {'Entry $':<8} {'Exit $':<8} "
              f"{'Stop Loss $':<11} {'Target $':<8} {'Shares':<6} {'P&L ($)':<10} "
              f"{'Result':<16} {'Regime':<14} Balance After$")
        print("  " + "-" * 145)

        for tr in t_trades:
            arrow   = "▲" if tr["P&L"] > 0 else "▼"
            regime  = tr.get("Regime", "?")
            print(
                f"  {tr['Entry Time']:<16}  {tr['Exit Time']:<16}  "
                f"${tr['Entry $']:<7.2f} ${tr['Exit $']:<7.2f} "
                f"${tr['Stop Loss $']:<10.2f} ${tr['Target $']:<7.2f} "
                f"{tr['Shares']:<6} {arrow} ${abs(tr['P&L']):<8.2f} "
                f"{tr['Result']:<16} {regime:<14} ${tr['Balance']:.2f}"
            )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
    