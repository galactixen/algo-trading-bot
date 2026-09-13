import yfinance as yf
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
load_dotenv()

def run_debug_session():
    ticker = "AAPL"
    # Testing over 2 days to see clear patterns
    df = yf.download(ticker, period="2d", interval="1m", progress=False)
    
    if df.empty: return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicators
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_S'] = macd['MACDs_12_26_9']

    # Simulation Variables
    in_position = False
    entry_price = 0
    trade_count = 0
    trade_history = []

    print(f"\n{'='*60}")
    print(f"MOCK SESSION: {ticker} 1-MINUTE SCALPING TRACE")
    print(f"{'='*60}")
    print(f"{'TIMESTAMP':<20} | {'ACTION':<7} | {'PRICE':<8} | {'PnL'}")
    print(f"{'-'*60}")

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        time_str = str(df.index[i])[11:16] 

        if not in_position:
            # ENTRY LOGIC: Price crosses VWAP + MACD Bullish Cross
            if (curr['Close'] > curr['VWAP'] and prev['Close'] <= prev['VWAP'] and curr['MACD'] > curr['MACD_S']):
                entry_price = float(curr['Close'])
                in_position = True
                trade_count += 1
                print(f"{time_str:<20} | BUY    | {entry_price:<8.2f} | ---")

        elif in_position:
            # EXIT LOGIC: 0.5% Target or 0.2% Stop
            profit_target = entry_price * 1.005
            stop_loss = entry_price * 0.998
            
            pnl_pct = ((curr['Close'] - entry_price) / entry_price) * 100
            
            if curr['High'] >= profit_target:
                print(f"{time_str:<20} | SELL   | {profit_target:<8.2f} | +0.50% (TARGET)")
                trade_history.append(0.50)
                in_position = False
            elif curr['Low'] <= stop_loss:
                print(f"{time_str:<20} | SELL   | {stop_loss:<8.2f} | -0.20% (STOP)")
                trade_history.append(-0.20)
                in_position = False

    # Summary
    if trade_history:
        win_rate = (len([x for x in trade_history if x > 0]) / len(trade_history)) * 100
        print(f"{'-'*60}")
        print(f"TOTAL TRADES: {trade_count} | WIN RATE: {win_rate:.2f}%")
        print(f"NET PERFORMANCE: {sum(trade_history):.2f}%")
    else:
        print("No trades executed with current parameters.")

if __name__ == "__main__":
    run_debug_session()