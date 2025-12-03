import tushare as ts
import pandas as pd

# Configuration
TUSHARE_TOKEN = "b0e7q171j0i329i258"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(TUSHARE_TOKEN)
pro._DataApi__token = TUSHARE_TOKEN
pro._DataApi__http_url = 'http://pro.tushare.nlink.vip'

def test_fina_indicator_with_codes():
    print("\n" + "="*30)
    print("Testing fina_indicator with 2000 codes...")
    try:
        # First get some codes
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
        codes = df_basic['ts_code'].tolist()[:2000] # Take first 2000
        print(f"Prepared {len(codes)} codes.")
        
        # Try to fetch
        df = pro.fina_indicator(ts_code=",".join(codes), period='20231231', fields='ts_code,dt_netprofit_yoy')
        print(f"Successfully fetched {len(df)} rows.")
        
    except Exception as e:
        print(f"Error fetching fina_indicator with codes: {e}")

if __name__ == "__main__":
    test_fina_indicator_with_codes()
