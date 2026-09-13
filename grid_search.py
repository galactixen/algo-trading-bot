import yfinance as yf
import pandas as pd
import pandas_ta as ta
import itertools
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------
# 1. DATA PREPARATION 
# ---------------------------------------------------------
def prepare_data(ticker):
    print(f"[{ticker}] Downloading and calculating indicators...")
    df_1m = yf.download(ticker, period="7d", interval="1m", progress=False)
    df_5m = yf.download(ticker, period="7d", interval="5m", progress=False)
    
    if df_1m.empty or df_5m.empty: 
        return None

    if isinstance(df_1m.columns, pd.MultiIndex):
        df_1m.columns = df_1m.columns.get_level_values(0)
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    df_1m.index = df_1m.index.tz_convert('US/Eastern')
    df_5m.index = df_5m.index.tz_convert('US/Eastern')

    df_5m['EMA50_5m'] = ta.ema(df_5m['Close'], length=50)
    df_1m['EMA50_5m'] = df_5m['EMA50_5m'].reindex(df_1m.index, method='ffill')

    df_1m['VWAP'] = ta.vwap(df_1m['High'], df_1m['Low'], df_1m['Close'], df_1m['Volume'])
    df_1m['EMA50'] = ta.ema(df_1m['Close'], length=50)
    macd = ta.macd(df_1m['Close'])
    df_1m['MACD'] = macd['MACD_12_26_9']
    df_1m['MACD_S'] = macd['MACDs_12_26_9']
    df_1m['ATR'] = ta.atr(df_1m['High'], df_1m['Low'], df_1m['Close'], length=14)
    
    return df_1m

# ---------------------------------------------------------
# 2. THE CORE BACKTEST ENGINE (Runs instantly in memory)
# ---------------------------------------------------------
def run_simulation(df_1m, stop_loss_mult, take_profit_mult, cooldown_mins, starting_balance=10000.0):
    balance = starting_balance
    in_position = False
    entry_price = 0
    shares = 0
    remainder = 0
    trade_history = []
    
    last_exit_time = None
    risk_per_trade_pct = 0.01 
    slippage_fee_rate = 0.0005 

    for i in range(50, len(df_1m)): 
        curr = df_1m.iloc[i]
        dt = df_1m.index[i]
        
        is_morning = (dt.hour == 9 and dt.minute >= 45) or (dt.hour == 10) or (dt.hour == 11 and dt.minute <= 30)
        is_eod = (dt.hour == 15 and dt.minute >= 55)

        if not in_position:
            if pd.isna(curr['EMA50_5m']) or pd.isna(curr['ATR']): continue

            cooldown_cleared = True
            if last_exit_time and (dt - last_exit_time).total_seconds() < (cooldown_mins * 60):
                cooldown_cleared = False
            
            above_vwap = curr['Close'] > curr['VWAP']
            touching_ema = curr['Low'] <= (curr['EMA50'] * 1.0015)
            holding_ema = curr['Close'] > curr['EMA50']
            green_candle = curr['Close'] > curr['Open']
            htf_bullish = curr['Close'] > curr['EMA50_5m']
            macd_bullish = curr['MACD'] > curr['MACD_S']

            if (is_morning and cooldown_cleared and above_vwap and 
                touching_ema and holding_ema and green_candle and 
                htf_bullish and macd_bullish):
                
                stop_loss_dist = curr['ATR'] * stop_loss_mult
                ideal_shares = int((balance * risk_per_trade_pct) // stop_loss_dist)
                entry_price = float(curr['Close']) * (1 + slippage_fee_rate)
                max_affordable = int(balance // entry_price)
                shares = min(ideal_shares, max_affordable) 
                
                if shares <= 0: continue
                
                invested_amount = shares * entry_price
                remainder = balance - invested_amount
                stop_loss = entry_price - stop_loss_dist
                profit_target = entry_price + (curr['ATR'] * take_profit_mult)
                in_position = True

        elif in_position:
            exit_price = 0
            if curr['High'] >= profit_target: exit_price = profit_target
            elif curr['Low'] <= stop_loss: exit_price = stop_loss
            elif is_eod: exit_price = float(curr['Close'])

            if exit_price > 0:
                exit_price = exit_price * (1 - slippage_fee_rate) 
                balance = remainder + (shares * exit_price) 
                in_position = False
                last_exit_time = dt
                
                if exit_price > entry_price: trade_history.append(1)
                else: trade_history.append(-1)

    net_profit = balance - starting_balance
    win_rate = (trade_history.count(1) / len(trade_history) * 100) if trade_history else 0
    return net_profit, len(trade_history), win_rate

# ---------------------------------------------------------
# 3. THE GRID SEARCH OPTIMIZER
# ---------------------------------------------------------
def optimize_portfolio():
    watchlist = ["AAPL", "NVDA", "TSLA", "AMD", "META"]
    
    sl_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    tp_multipliers = [1.5, 2.0, 3.0, 4.0, 5.0]
    cooldowns_mins = [0, 15, 30, 60]
    
    combinations = list(itertools.product(sl_multipliers, tp_multipliers, cooldowns_mins))
    
    print(f"\n{'='*75}")
    print(f"RUNNING ALGORITHMIC GRID SEARCH ({len(combinations)} Combos Per Ticker)")
    print(f"{'='*75}\n")
    
    optimized_dna = {}

    for ticker in watchlist:
        df_1m = prepare_data(ticker)
        if df_1m is None: continue
        
        best_profit = -float('inf')
        best_combo = None
        best_stats = {}
        
        for sl, tp, cooldown in combinations:
            profit, trades, win_rate = run_simulation(df_1m, sl, tp, cooldown)
            
            if profit > best_profit and trades > 0:
                best_profit = profit
                best_combo = (sl, tp, cooldown)
                best_stats = {"Trades": trades, "Win Rate": win_rate, "Profit": profit}
                
        if best_combo:
            optimized_dna[ticker] = {
                "stop_loss_mult": best_combo[0],
                "take_profit_mult": best_combo[1],
                "cooldown_mins": best_combo[2]
            }
            print(f"--> [WINNER] {ticker}: SL={best_combo[0]}x | TP={best_combo[1]}x | Cooldown={best_combo[2]}m")
            print(f"    Results: {best_stats['Trades']} Trades | {best_stats['Win Rate']:.1f}% Win Rate | ${best_stats['Profit']:.2f} Profit\n")
        else:
            print(f"--> {ticker}: No profitable combinations found for this timeframe.\n")

    print(f"{'='*75}")
    print("COPY AND PASTE THIS DICTIONARY INTO YOUR MASTER BOT:")
    print(f"{'='*75}")
    print("ticker_dna = {")
    for tick, dna in optimized_dna.items():
        print(f"    '{tick}': {dna},")
    print("}")

if __name__ == "__main__":
    optimize_portfolio()