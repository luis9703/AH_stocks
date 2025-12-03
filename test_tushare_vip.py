import tushare as ts
import pandas as pd
import time

# Configuration
TUSHARE_TOKEN = "b0e7q171j0i329i258"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(TUSHARE_TOKEN)
pro._DataApi__token = TUSHARE_TOKEN
pro._DataApi__http_url = 'http://pro.tushare.nlink.vip'

def test_daily_basic():
    print("="*30)
    print("Testing daily_basic for 20240515...")
    try:
        # Try to fetch all stocks for a specific date
        # 20240515 is a trading day
        df = pro.daily_basic(ts_code='', trade_date='20240515', fields='ts_code,trade_date,pe_ttm,total_mv')
        print(f"Successfully fetched {len(df)} rows.")
        if not df.empty:
            print("Sample data:")
            print(df.head())
            print(f"Unique dates: {df['trade_date'].unique()}")
            
            # Check if we have data for many stocks (e.g. > 4000)
            if len(df) > 4000:
                print("SUCCESS: Retrieved data for > 4000 stocks.")
            else:
                print(f"WARNING: Only retrieved {len(df)} stocks. Might be incomplete.")
        else:
            print("Returned empty DataFrame.")
    except Exception as e:
        print(f"Error fetching daily_basic: {e}")

def test_fina_indicator():
    print("\n" + "="*30)
    print("Testing fina_indicator for period 20240331...")
    try:
        # Try to fetch financial indicators for all stocks for a specific period
        # Tushare usually limits rows (e.g. 2000 or 5000). 
        # Let's see how many we get.
        df = pro.fina_indicator(period='20240331', fields='ts_code,end_date,dt_netprofit_yoy')
        print(f"Successfully fetched {len(df)} rows.")
        if not df.empty:
            print("Sample data:")
            print(df.head())
            
            # Check coverage
            if len(df) > 4000:
                print("SUCCESS: Retrieved financial data for > 4000 stocks in one go.")
            else:
                print(f"OBSERVATION: Retrieved {len(df)} stocks. If this is exactly 2000/4000/5000, it hit a limit.")
        else:
            print("Returned empty DataFrame.")
            
    except Exception as e:
        print(f"Error fetching fina_indicator: {e}")

def test_stock_basic():
    print("\n" + "="*30)
    print("Testing stock_basic...")
    try:
        df = pro.stock_basic(exchange='', list_status='L,D,P', fields='ts_code,name,market,list_status')
        print(f"Successfully fetched {len(df)} rows.")
        if not df.empty:
            print("Sample data:")
            print(df.head())
    except Exception as e:
        print(f"Error fetching stock_basic: {e}")

if __name__ == "__main__":
    test_daily_basic()
    test_fina_indicator()
    test_stock_basic()
