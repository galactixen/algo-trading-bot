import yfinance as yf
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
load_dotenv()

def run_aapl_max_test():
    ticker = "AAPL"
    print(f"--- FETCHING 2 YEARS OF HISTORICAL DATA FOR {ticker} ---")
    
    # 1-hour interval allows for a 730-day (2 year) lookback
    df = yf.download(ticker, period="730d", interval="1h", progress=False)
    
    if df.empty: 
        print("No data found.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- INDICATORS ---
    df['Signal'] = 0
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['BBL'] = bbands.iloc[:, 0]
    
    # Volume Surge (Comparing to 20-hour average)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()

    # --- SIMULATION ---
    in_position = False
    entry_price = 0
    highest_price = 0
    trade_count = 0
    
    # RISK PARAMETERS
    stop_loss_pct = 0.02      # 2% Hard Stop (Safety net)
    trailing_pct = 0.015      # 1.5% Trailing Stop (Profit lock)

    for i in range(len(df)):
        if i < 200 or pd.isna(df['BBL'].iloc[i]): continue
        
        curr = df.iloc[i]
        
        if not in_position:
            # OPTIMIZED AAPL ENTRY: RSI < 40 
            if (curr['RSI'] < 40 and 
                curr['Close'] > curr['EMA200'] and 
                curr['Close'] <= curr['BBL'] and 
                curr['Volume'] > curr['Vol_Avg']):
                
                df.at[df.index[i], 'Signal'] = 1
                entry_price = float(curr['Close'])
                highest_price = entry_price
                in_position = True
                trade_count += 1

        elif in_position:
            # Update peak price for trailing stop
            if curr['Close'] > highest_price:
                highest_price = float(curr['Close'])

            # EXIT LOGIC
            trailing_exit = highest_price * (1 - trailing_pct)
            hard_stop_exit = entry_price * (1 - stop_loss_pct)

            if curr['Close'] <= trailing_exit:
                df.at[df.index[i], 'Signal'] = -1
                in_position = False
            elif curr['Close'] <= hard_stop_exit:
                df.at[df.index[i], 'Signal'] = -1
                in_position = False
            elif curr['RSI'] > 75: 
                df.at[df.index[i], 'Signal'] = -1
                in_position = False

    # --- RESULTS ---
    df['Pct_Change'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Pct_Change']
    
    total_return = (df['Strategy_Return'] + 1).prod()
    buy_and_hold = (df['Pct_Change'] + 1).prod()

    print(f"\nRESULTS FOR {ticker} OVER 2 YEARS:")
    print(f"{'='*30}")
    print(f"Total Bot Return:    {(total_return - 1) * 100:.2f}%")
    print(f"Buy & Hold Return:   {(buy_and_hold - 1) * 100:.2f}%")
    print(f"Total Trades:        {trade_count}")
    print(f"{'='*30}")

if __name__ == "__main__":
    run_aapl_max_test()