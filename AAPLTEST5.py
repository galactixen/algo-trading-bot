import os
import time
import logging
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import pandas_ta as ta
import pytz

# ── Alpaca SDK ────────────────────────────────────────────────────────────────
from alpaca.trading.client          import TradingClient
from alpaca.trading.requests        import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums           import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical         import StockHistoricalDataClient
from alpaca.data.requests           import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe          import TimeFrame, TimeFrameUnit

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

COMMISSION         = 0.0
RISK_PCT           = 0.01
MAX_POSITION_PCT   = 0.05   
SL_MULT            = 2.0
TP_MULT            = 4.0
MAX_DAILY_LOSS     = 150.0
MACD_LOOKBACK_BARS = 12
PULLBACK_BAND      = 0.004
TOP_N_TICKERS      = 5

# --- NEW CONFIG VARIABLES ---
MIN_TREND_SCORE         = 30.0  # Minimum trend score required to trade a ticker
MAX_SLIPPAGE_CENTS      = 0.05  # Max cents willing to pay over the Ask price
ORDER_LIFESPAN_SECS     = 15    # Seconds to wait for a fill before cancelling entry order

MAX_TRADES_PER_DAY      = 3     
MAX_CONSECUTIVE_LOSSES  = 3     
FILL_POLL_TIMEOUT_SECS  = 45    
API_RETRY_ATTEMPTS      = 3     
API_RETRY_DELAY_SECS    = 2     

ALL_TICKERS = ["TSLA", "NVDA", "AMD", "META", "AAPL",
               "MSFT", "AMZN", "GOOGL", "AVGO", "LLY"]

EASTERN = pytz.timezone("US/Eastern")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ALPACA CLIENTS  (initialised once at startup)
# ─────────────────────────────────────────────
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def api_call_with_retry(fn, *args, label: str = "", **kwargs):
    last_exc = None
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            log.warning(f"  API call '{label or fn.__name__}' failed (attempt {attempt}/{API_RETRY_ATTEMPTS}): {exc}")
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECS)
    raise last_exc

# ─────────────────────────────────────────────
# HELPERS — ACCOUNT / POSITION STATE
# ─────────────────────────────────────────────

def get_account_balance() -> float:
    account = api_call_with_retry(trading_client.get_account, label="get_account")
    return float(account.equity)

def get_open_positions() -> dict:
    positions = api_call_with_retry(trading_client.get_all_positions, label="get_all_positions")
    return {p.symbol: p for p in positions}

def cancel_all_orders_for(ticker: str):
    req    = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
    orders = api_call_with_retry(trading_client.get_orders, req, label="get_orders")
    for o in orders:
        try:
            trading_client.cancel_order_by_id(o.id)
            log.info(f"Cancelled open order {o.id} for {ticker}")
        except Exception as e:
            log.warning(f"Could not cancel order {o.id}: {e}")

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_bars(ticker: str, days: int, timeframe: TimeFrame) -> pd.DataFrame:
    start = datetime.now(EASTERN) - timedelta(days=days)
    req   = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=timeframe,
        start=start,
    )
    bars = api_call_with_retry(data_client.get_stock_bars, req, label=f"get_stock_bars:{ticker}").df

    if bars.empty: return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")

    bars.index = bars.index.tz_convert(EASTERN)
    bars.columns = [c.capitalize() for c in bars.columns]
    return bars

def calculate_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = df.index.date
    vwap_series = (
        df.groupby("Date", group_keys=False)
          .apply(lambda g: ta.vwap(g["High"], g["Low"], g["Close"], g["Volume"]))
    )
    df["VWAP"] = vwap_series.values
    return df.drop(columns="Date")

def build_intraday_df(ticker: str) -> pd.DataFrame | None:
    df = fetch_bars(ticker, days=60, timeframe=TimeFrame(5, TimeFrameUnit.Minute))
    if df.empty: return None

    df = calculate_daily_vwap(df)
    df["EMA50"]    = ta.ema(df["Close"], length=50)

    macd           = ta.macd(df["Close"])
    df["MACD"]     = macd["MACD_12_26_9"]
    df["MACD_S"]   = macd["MACDs_12_26_9"]

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

def fetch_daily_trend(ticker: str) -> dict:
    df = fetch_bars(ticker, days=90, timeframe=TimeFrame.Day)
    if df.empty: return {}
    df["EMA20_D"] = ta.ema(df["Close"], length=20)
    df.dropna(inplace=True)
    return {
        (dt.date() if hasattr(dt, "date") else dt): float(row["Close"]) > float(row["EMA20_D"])
        for dt, row in df.iterrows()
    }

def prior_day_bullish(daily_trend: dict, today: date) -> bool:
    for days_back in range(1, 6):
        prior = today - timedelta(days=days_back)
        if prior in daily_trend: return daily_trend[prior]
    return False

# ─────────────────────────────────────────────
# TREND QUALITY SCORER (UPDATED)
# ─────────────────────────────────────────────

def score_trend_quality(df: pd.DataFrame) -> float:
    recent = df.tail(200).copy()
    if len(recent) < 50: return 0.0
    adx_score      = float(recent["ADX"].mean())
    di_score       = float((recent["DI_plus"] - recent["DI_minus"]).mean())
    above_vwap_pct = float((recent["Close"] > recent["VWAP"]).mean()) * 100
    ema_start      = float(recent["EMA50"].iloc[0])
    ema_end        = float(recent["EMA50"].iloc[-1])
    ema_slope      = ((ema_end - ema_start) / ema_start) * 10_000
    return round(adx_score * 0.40 + di_score * 0.30 + above_vwap_pct * 0.20 + ema_slope * 0.10, 2)

def select_top_tickers(tickers: list, top_n: int = TOP_N_TICKERS):
    log.info("Scoring ticker trend quality...")
    scores = {}
    for ticker in tickers:
        df = build_intraday_df(ticker)
        if df is None: continue
        score = score_trend_quality(df)
        scores[ticker] = score

    # Sort highest to lowest
    all_scores  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Absolute Gatekeeper: Only accept tickers that meet MIN_TREND_SCORE
    top_tickers = [t for t, score in all_scores if score >= MIN_TREND_SCORE][:top_n]
    
    if len(top_tickers) < top_n:
        log.warning(f"Only found {len(top_tickers)} tickers meeting the minimum trend score of {MIN_TREND_SCORE}.")
        
    return top_tickers

# ─────────────────────────────────────────────
# EXECUTION & POLLING
# ─────────────────────────────────────────────

def wait_for_fill(order_id: str, timeout_secs: int = FILL_POLL_TIMEOUT_SECS) -> float | None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            if order.status == "filled":
                fill_price = float(order.filled_avg_price)
                log.info(f"  Order {order_id} filled @ ${fill_price:.4f}")
                return fill_price
            if order.status in ("canceled", "expired", "rejected"):
                log.warning(f"  Order {order_id} ended with status: {order.status}")
                return None
        except Exception:
            pass
        time.sleep(1)

    log.warning(f"  Order {order_id} not fully filled within {timeout_secs}s — cancelling remainder.")
    try: trading_client.cancel_order_by_id(order_id)
    except: pass
    return None

def submit_bracket_order(ticker: str, shares: int, limit_price: float, sl_price: float, tp_price: float):
    try:
        req = LimitOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            stop_loss=StopLossRequest(
                stop_price=round(sl_price, 2)
            )
        )
        order = trading_client.submit_order(req)
        log.info(f"  Bracket submitted: BUY {shares}x {ticker} @ Marketable LMT {limit_price:.2f}")
        return order
    except Exception as e:
        log.error(f"  Order failed for {ticker}: {e}")
        return None

def close_position(ticker: str, reason: str):
    cancel_all_orders_for(ticker)  
    positions = get_open_positions()
    if ticker not in positions: return

    shares = int(float(positions[ticker].qty))
    log.info(f"  [{ticker}] Closing {shares}sh — reason: {reason}")
    req = MarketOrderRequest(symbol=ticker, qty=shares, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    try:
        order = trading_client.submit_order(req)
        wait_for_fill(order.id)
    except Exception as e:
        log.error(f"  Failed to market-close {ticker}: {e}")

# ─────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────

def check_entry_signal(df: pd.DataFrame, bars_since_cross: int | None) -> tuple[bool, int | None]:
    df = df.iloc[:-1]
    if len(df) < 3: return False, bars_since_cross

    curr, prev = df.iloc[-1], df.iloc[-2]

    if float(curr["ADX"]) < 25: 
        return False, bars_since_cross
    if float(curr["Volume"]) < float(curr["Vol_MA20"]) * 1.2:
        return False, bars_since_cross

    macd_up   = (prev["MACD"] <= prev["MACD_S"]) and (curr["MACD"] > curr["MACD_S"])
    macd_down = (prev["MACD"] >= prev["MACD_S"]) and (curr["MACD"] < curr["MACD_S"])

    if macd_up: bars_since_cross = 0
    elif macd_down: bars_since_cross = None
    elif bars_since_cross is not None:
        bars_since_cross += 1
        if bars_since_cross > MACD_LOOKBACK_BARS: bars_since_cross = None

    in_window = bars_since_cross is not None and bars_since_cross >= 1
    near_vwap = 0 <= (float(curr["Close"]) - float(curr["VWAP"]))  / float(curr["VWAP"])  < PULLBACK_BAND
    near_ema  = 0 <= (float(curr["Close"]) - float(curr["EMA50"])) / float(curr["EMA50"]) < PULLBACK_BAND

    signal = in_window and (near_vwap or near_ema) and (float(curr["Close"]) > float(curr["VWAP"])) and (float(curr["Close"]) > float(curr["EMA50"])) and (float(curr["DI_plus"]) > float(curr["DI_minus"]))
    return signal, bars_since_cross

def seconds_to_next_bar(now: datetime, buffer_secs: int = 5) -> int:
    seconds_into_5min = (now.minute % 5) * 60 + now.second
    raw_sleep = 300 - seconds_into_5min + buffer_secs
    return max(raw_sleep, 5)

# ─────────────────────────────────────────────
# SESSION GUARD
# ─────────────────────────────────────────────

class SessionGuard:
    def __init__(self, start_equity: float):
        self.start_equity       = start_equity
        self.trades_today       = 0
        self._halted            = False

    def record_trade_open(self):
        self.trades_today += 1

    def allow_new_entry(self, current_equity: float) -> bool:
        if self._halted: return False
        
        if self.start_equity - current_equity >= MAX_DAILY_LOSS:
            log.warning(f"  MAX DAILY LOSS HIT. Starting Equity: {self.start_equity}, Current: {current_equity}")
            self._halted = True
            return False
            
        if self.trades_today >= MAX_TRADES_PER_DAY: return False
        return True

# ─────────────────────────────────────────────
# MAIN LIVE TRADING LOOP (UPDATED)
# ─────────────────────────────────────────────

def is_market_open() -> bool:
    clock = api_call_with_retry(trading_client.get_clock, label="get_clock")
    return clock.is_open

def market_hours_check(now: datetime) -> tuple[bool, bool]:
    h, m   = now.hour, now.minute
    active = (h == 10 and m >= 15) or (10 < h < 15) or (h == 15 and m < 30)
    eod    = h == 15 and m >= 30 
    return active, eod

def run_live():
    log.info("=" * 60)
    log.info("ALPACA LIVE TRADING — MOMENTUM STRATEGY (V4 STABLE)")
    log.info("=" * 60)

    while not is_market_open(): time.sleep(60)

    top_tickers = select_top_tickers(ALL_TICKERS, TOP_N_TICKERS)
    bars_since_cross: dict[str, int | None] = {t: None for t in top_tickers}
    
    session_start_equity = get_account_balance()
    guard = SessionGuard(start_equity=session_start_equity)
    
    pending_orders = {} 

    while True:
        now         = datetime.now(EASTERN)
        active, eod = market_hours_check(now)

        if eod:
            log.info("EOD (3:30 PM Trigger) — closing all positions.")
            open_positions = get_open_positions()
            for ticker in open_positions.keys():
                close_position(ticker, "EOD Exit")
            break

        if not active:
            time.sleep(60)
            continue

        try: current_equity = get_account_balance()
        except: time.sleep(60); continue

        # ── Non-Blocking Order Tracking ──
        for ticker, order_info in list(pending_orders.items()):
            try:
                order = trading_client.get_order_by_id(order_info["order_id"])
                if order.status == "filled":
                    log.info(f"  [FILLED] Order {order.id} for {ticker}")
                    guard.record_trade_open()
                    del pending_orders[ticker]
                elif order.status in ("canceled", "expired", "rejected"):
                    log.warning(f"  [CLOSED] Order {order.id} for {ticker} ended with status: {order.status}")
                    del pending_orders[ticker]
                elif time.time() - order_info["timestamp"] > ORDER_LIFESPAN_SECS:
                    log.warning(f"  [TIMEOUT] Order {order.id} for {ticker} unfilled after {ORDER_LIFESPAN_SECS}s. Cancelling to chase.")
                    trading_client.cancel_order_by_id(order.id)
                    del pending_orders[ticker]
            except Exception as e:
                log.error(f"  Error checking pending order for {ticker}: {e}")

        open_positions = get_open_positions()

        for ticker in top_tickers:
            
            # Skip if we already own it, or if we have an entry order actively trying to fill
            if ticker in open_positions or ticker in pending_orders:
                continue

            if not guard.allow_new_entry(current_equity): 
                continue
                
            try:
                df          = build_intraday_df(ticker)
                daily_trend = fetch_daily_trend(ticker)
            except: continue

            if df is None or df.empty: continue

            completed = df.iloc[-2]
            close = float(completed["Close"])
            atr   = float(completed["ATR"])

            if not prior_day_bullish(daily_trend, now.date()): continue

            signal, bars_since_cross[ticker] = check_entry_signal(df, bars_since_cross[ticker])

            if signal:
                # ── Real-Time Execution Logic ──
                try:
                    quote_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                    latest_quote = data_client.get_stock_latest_quote(quote_req)[ticker]
                    real_time_ask = float(latest_quote.ask_price)
                except Exception as e:
                    log.error(f"Failed to fetch live quote for {ticker}: {e}")
                    continue
                    
                if real_time_ask == 0.0:
                    real_time_ask = close

                limit_price = real_time_ask + MAX_SLIPPAGE_CENTS

                atr_pct = atr / close
                capped_atr = min(max(atr_pct, 0.005), 0.03) * real_time_ask 

                sl_dist     = capped_atr * SL_MULT
                tp_dist     = capped_atr * TP_MULT
                
                ideal_shares = int((current_equity * RISK_PCT) / sl_dist)
                max_allowed_shares = int((current_equity * MAX_POSITION_PCT) / limit_price)
                shares = min(ideal_shares, max_allowed_shares, int(current_equity // limit_price))

                if shares <= 0: continue

                stop_loss_px   = limit_price - sl_dist
                take_profit_px = limit_price + tp_dist

                order = submit_bracket_order(ticker, shares, limit_price, stop_loss_px, take_profit_px)
                if order is None: continue

                # Register as pending instead of blocking the loop
                pending_orders[ticker] = {
                    "order_id": order.id,
                    "timestamp": time.time()
                }
                bars_since_cross[ticker] = None

        now        = datetime.now(EASTERN)
        sleep_secs = seconds_to_next_bar(now, buffer_secs=5)
        
        if pending_orders:
            time.sleep(2) 
        else:
            time.sleep(sleep_secs)

if __name__ == "__main__":
    run_live()