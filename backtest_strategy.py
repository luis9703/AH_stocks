import tushare as ts
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import time
import warnings

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# 禁用代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['all_proxy'] = ''

# --- Configuration ---
# Tushare Token配置 (VIP接口)
TUSHARE_TOKEN = "b0e7q171j0i329i258"
ts.set_token(TUSHARE_TOKEN)
# 设置超时时间为30秒
pro = ts.pro_api(TUSHARE_TOKEN, timeout=30)
pro._DataApi__token = TUSHARE_TOKEN
pro._DataApi__http_url = 'http://pro.tushare.nlink.vip'

# Backtest Parameters (回测参数)
# 回测时间范围：过去6年到今天
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365*6)).strftime('%Y%m%d')
END_DATE = datetime.datetime.now().strftime('%Y%m%d')
INITIAL_CASH = 1000000       # 初始资金 100万
MAX_POSITIONS = 5           # 最大持仓股票数量
POSITION_SIZE_PCT = 0.20     # 单只股票仓位占比 20%
HOLD_DAYS = 20               # 持仓天数

# Stock Filter Parameters (选股过滤参数)
FILTER_MIN_MARKET_CAP = 600 * 10000 * 10000  # 最小市值 350亿
FILTER_MIN_PE = 15           # 最小市盈率(TTM) - 下调至3以包含银行等低估值蓝筹
FILTER_MAX_PE = 45           # 最大市盈率(TTM)
FILTER_MIN_PROFIT_YOY = 15  # 净利润同比增长率下限 (%)
# FILTER_FINANCIAL_PERIOD removed - will be dynamic (财务周期动态获取)

# Strategy Parameters (策略参数)
# 买入评分权重：主力吸货(40%) + 风险控制(35%) + 动量涨跌(25%)
BUY_WEIGHTS = {'s1_main': 0.30, 's2_risk': 0.30, 's3_momentum': 0.40}
BUY_THRESHOLD = 90           # 买入总分阈值
STOP_LOSS_PCT = -0.05        # 止损百分比 (-12%)
TAKE_PROFIT_PCT = 0.25       # 止盈百分比 (30%)
COOLDOWN_DAYS = 14           # 卖出后冷却天数 (禁止买入刚卖出的股票)

# Transaction Cost Parameters (交易成本参数)
COMMISSION_RATE = 0.0003     # 买卖佣金 0.03%
STAMP_DUTY_RATE = 0.001      # 卖出印花税 0.1%
SLIPPAGE_RATE = 0.002        # 滑点 0.2% (双向)

# Chart Parameters (图表参数)
CHART_FONT_SIZE_TITLE = 20
CHART_FONT_SIZE_LABEL = 15
CHART_FONT_SIZE_TICK = 12
CHART_FONT_SIZE_LEGEND = 12
CHART_DRAWDOWN_COLOR = 'red'
CHART_DRAWDOWN_ALPHA = 0.3

# Risk Metrics Parameters (风险指标参数)
RISK_FREE_RATE = 0.02        # 无风险利率 2%
REPO_RATE = 0.014            # 国债逆回购年化收益率 1.4%

# --- Helper Functions (Copied from stock_scanner.py) ---

def sma_cn(series, n, m):
    """
    模拟同花顺SMA算法: Y = (M*X + (N-M)*Y')/N
    用于计算平滑移动平均
    """
    return series.ewm(alpha=m/n, adjust=True).mean()

def calculate_signals_vectorized(df):
    """
    向量化计算所有信号
    返回包含 'signal' (True/False) 和 'score' 的DataFrame
    """
    if df is None or len(df) < 90:
        return df

    try:
        df = df.copy()
        # 统一列名
        if 'close' in df.columns:
            df.rename(columns={'close': 'C', 'low': 'L', 'high': 'H', 'open': 'O'}, inplace=True)
        
        df = df.sort_values('trade_date').reset_index(drop=True)

        # --- Indicator Calculation (指标计算) ---
        # 1. 主力吸货指标
        VAR1 = df['L'].shift(1)
        sma_abs = sma_cn(abs(df['L'] - VAR1), 3, 1)
        sma_max = sma_cn(np.maximum(df['L'] - VAR1, 0), 3, 1)
        VAR2 = (sma_abs / sma_max) * 100
        VAR2.fillna(0, inplace=True)
        VAR3 = (VAR2 * 10).ewm(span=3, adjust=False).mean()
        VAR4 = df['L'].rolling(38).min()
        VAR5 = VAR3.rolling(38).max()
        VAR6 = 1 # 简化处理
        
        condition_var7 = df['L'] <= VAR4
        val_if_true = (VAR3 + VAR5 * 2) / 2
        ema_base = np.where(condition_var7, val_if_true, 0)
        VAR7 = pd.Series(ema_base, index=df.index).ewm(span=3, adjust=False).mean() / 618 * VAR6
        df['主力吸货'] = VAR7
        
        # 2. 风险指标
        llv_21 = df['L'].rolling(21).min()
        hhv_21 = df['H'].rolling(21).max()
        range_21 = hhv_21 - llv_21
        VAR8 = np.where(range_21 > 0, (df['C'] - llv_21) / range_21 * 100, 0)
        VAR9 = sma_cn(pd.Series(VAR8, index=df.index), 13, 8)
        df['风险'] = np.ceil(sma_cn(VAR9, 13, 8))
        
        # 3. 涨跌动量指标
        llv_27 = df['L'].rolling(27).min()
        hhv_27 = df['H'].rolling(27).max()
        range_27 = hhv_27 - llv_27
        k_stoch_27 = np.where(range_27 > 0, (df['C'] - llv_27) / range_27 * 100, 0)
        sma1 = sma_cn(pd.Series(k_stoch_27, index=df.index), 5, 1)
        sma2 = sma_cn(sma1, 3, 1)
        inner_calc = 3 * sma1 - 2 * sma2
        df['涨跌'] = inner_calc.rolling(5).mean()
        
        # --- Scoring (评分系统) ---
        # 1. 主力吸货得分
        score1 = np.zeros(len(df))
        base_cond = df['主力吸货'] > 0
        cont_cond = (df['主力吸货'] > df['主力吸货'].shift(1)) & (df['主力吸货'].shift(1) > df['主力吸货'].shift(2))
        strength_cond = df['主力吸货'] > df['主力吸货'].rolling(20).mean()
        score1[base_cond] += 50
        score1[cont_cond] += 30
        score1[strength_cond] += 20
        df['主力吸货分数'] = score1
        
        # 2. 风险得分 (风险越低分越高)
        score2 = np.zeros(len(df))
        score2[df['风险'] < 20] = 100
        score2[(df['风险'] >= 20) & (df['风险'] < 50)] = 70
        score2[(df['风险'] >= 50) & (df['风险'] < 80)] = 30
        df['风险分数'] = score2
        
        # 3. 涨跌得分 (趋势向上分高)
        score3 = np.zeros(len(df))
        score3[(df['涨跌'] > 0) & (df['涨跌'] > df['涨跌'].shift(1))] = 100
        score3[(df['涨跌'] > 0) & (df['涨跌'] <= df['涨跌'].shift(1))] = 50
        df['涨跌分数'] = score3
        
        # 计算总分
        df['买入总分'] = (df['主力吸货分数'] * BUY_WEIGHTS['s1_main'] + 
                         df['风险分数'] * BUY_WEIGHTS['s2_risk'] + 
                         df['涨跌分数'] * BUY_WEIGHTS['s3_momentum'])
        
        # 生成买入信号
        df['signal'] = df['买入总分'] >= BUY_THRESHOLD
        return df

    except Exception as e:
        # print(f"Error: {e}")
        return df

# --- Data Fetching ---

def get_rebalance_schedule(start_date, end_date):
    """
    生成调仓日期表及对应的财报周期。
    规则:
    - 5月15日: 使用一季报 (0331)
    - 8月31日: 使用中报 (0630)
    - 10月31日: 使用三季报 (0930)
    """
    s_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    e_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    
    # 向前追溯一年以确保有初始股票池
    curr = s_date - datetime.timedelta(days=365)
    
    schedule = []
    while curr <= e_date:
        year = curr.year
        
        # 定义每年的调仓检查点
        checkpoints = [
            (datetime.datetime(year, 5, 15), f"{year}0331"),
            (datetime.datetime(year, 8, 31), f"{year}0630"),
            (datetime.datetime(year, 10, 31), f"{year}0930")
        ]
        
        for date_obj, period in checkpoints:
            # 收集所有可能相关的检查点
            schedule.append({'date': date_obj.strftime('%Y%m%d'), 'period': period})
            
        curr = datetime.datetime(year + 1, 1, 1)
        
    # 按日期排序
    schedule.sort(key=lambda x: x['date'])
    return schedule

def get_stock_pool(target_date, period):
    """
    获取特定历史日期的股票池（基于当时的基本面数据）。
    返回: (股票代码列表, {代码: 名称}字典)
    """
    print(f"Fetching stock pool for Rebalance Date: {target_date} (Period: {period})...")
    try:
        # 1. 找到目标日期或之前的最近交易日
        # Optimization: Look back 30 days instead of from 2010 to avoid data issues and improve speed
        dt_target = datetime.datetime.strptime(target_date, '%Y%m%d')
        dt_start = dt_target - datetime.timedelta(days=30)
        start_date_str = dt_start.strftime('%Y%m%d')
        
        cal = pro.trade_cal(exchange='', start_date=start_date_str, end_date=target_date, is_open='1')
        if cal.empty:
            print(f"Warning: No trade calendar found between {start_date_str} and {target_date}")
            return [], {}
        
        # Sort by date ascending to ensure iloc[-1] is the latest date
        cal = cal.sort_values('cal_date')
        last_trade_date = cal.iloc[-1]['cal_date']
        
        # Check for stale data
        last_trade_dt = datetime.datetime.strptime(last_trade_date, '%Y%m%d')
        if (dt_target - last_trade_dt).days > 10:
             print(f"  ! WARNING: Data might be stale! Last trade date {last_trade_date} is far from target {target_date}")

        # 2. 获取该日期的基础指标 (市值, PE)
        df_daily = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,pe_ttm,total_mv')
        if df_daily.empty:
             print(f"Warning: No daily basic data for {last_trade_date}")
             return [], {}
        
        # Debug: Print sample data to verify it changes
        print(f"  > Daily Basic Data for {last_trade_date}:")
        print(f"    Sample (000001.SZ): {df_daily[df_daily['ts_code']=='000001.SZ'][['pe_ttm', 'total_mv']].to_dict('records')}")
        print(f"    Total rows: {len(df_daily)}")

        # Fetch market info to ensure coverage of Main, ChiNext, STAR
        # CRITICAL FIX: Include Delisted (D) and Paused (P) stocks to avoid Survivorship Bias
        # 关键修正：包含已退市(D)和暂停上市(P)的股票，避免幸存者偏差
        df_basic = pro.stock_basic(exchange='', list_status='L,D,P', fields='ts_code,name,market')
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # 过滤: 市值 & PE
        subset = df[
            (df['total_mv'] * 10000 >= FILTER_MIN_MARKET_CAP) & 
            (df['pe_ttm'] >= FILTER_MIN_PE) & 
            (df['pe_ttm'] <= FILTER_MAX_PE)
        ].copy()
        
        print(f"  > After Market Cap & PE filter: {len(subset)}")
        # Print Market Distribution
        market_counts = subset['market'].value_counts()
        print(f"  > Market Distribution: {market_counts.to_dict()}")

        # 3. 财务过滤 (净利润增长率 > 0) 针对特定财报周期
        # Note: dt_netprofit_yoy in fina_indicator usually refers to Deducted Net Profit YoY (Accumulated/YTD for the period)
        # 这里的 dt_netprofit_yoy 指的是扣非净利润同比增长率 (期末累计值，即YTD)
        codes = subset['ts_code'].tolist()
        if not codes:
            return [], {}
            
        # 批量获取财务指标
        # Note: Tushare limit is 1000 codes per request. Chunk if necessary.
        df_fina_list = []
        chunk_size = 800 # Safe limit
        for i in range(0, len(codes), chunk_size):
            chunk_codes = codes[i:i+chunk_size]
            try:
                df_chunk = pro.fina_indicator(ts_code=",".join(chunk_codes), period=period, fields='ts_code,dt_netprofit_yoy')
                df_fina_list.append(df_chunk)
            except Exception as e:
                print(f"  ! Error fetching fina_indicator chunk {i}: {e}")
        
        if df_fina_list:
            df_fina = pd.concat(df_fina_list, ignore_index=True)
            # Deduplicate: Keep the latest record if multiple exist for same ts_code (though period is same)
            # Tushare might return multiple records if there are updates.
            # Usually we want the latest one, but here we don't have publish date.
            # We can just drop duplicates on ts_code.
            original_len = len(df_fina)
            df_fina.drop_duplicates(subset=['ts_code'], keep='last', inplace=True)
            if len(df_fina) < original_len:
                print(f"  > Deduplicated financial data: {original_len} -> {len(df_fina)}")
        else:
            df_fina = pd.DataFrame()
        
        if not df_fina.empty:
            # Debug: Print sample financial data
            print(f"  > Financial Data Sample (First 2):")
            print(df_fina[['ts_code', 'dt_netprofit_yoy']].head(2).to_string(index=False))
            
            subset = pd.merge(subset, df_fina, on='ts_code', how='inner')
            print(f"  > After merging financial data: {len(subset)}")
            subset = subset[subset['dt_netprofit_yoy'] > FILTER_MIN_PROFIT_YOY]
            print(f"  > After Profit YoY filter: {len(subset)}")
            
        # 去重
        subset.drop_duplicates(subset=['ts_code'], inplace=True)

        print(f"  > Final Pool size: {len(subset)}")
        # Return list of codes and dict of info (name, market)
        info_dict = {}
        for _, row in subset.iterrows():
            info_dict[row['ts_code']] = {'name': row['name'], 'market': row['market']}
            
        return subset['ts_code'].tolist(), info_dict
        
    except Exception as e:
        print(f"Error fetching pool for {target_date}: {e}")
        return [], {}
        
    except Exception as e:
        print(f"Error fetching pool for {target_date}: {e}")
        return [], {}

def fetch_all_data(stock_pool, start_date, end_date):
    """
    获取股票池中所有股票的日线数据。
    """
    data_map = {}
    print(f"Fetching data for {len(stock_pool)} stocks...")
    
    # 将开始日期前推，以确保指标计算有足够的预热数据 (例如 200天)
    warmup_start = (datetime.datetime.strptime(start_date, "%Y%m%d") - datetime.timedelta(days=200)).strftime("%Y%m%d")
    
    chunk_size = 5
    for i in range(0, len(stock_pool), chunk_size):
        chunk = stock_pool[i:i+chunk_size]
        print(f"Fetching chunk {i//chunk_size + 1}...")
        
        # Tushare pro_bar 接口通常用于单只股票获取复权数据
        for code in chunk:
            for attempt in range(3): # Retry logic
                try:
                    # Pass the configured 'pro' API object to ensure VIP URL is used
                    df = ts.pro_bar(ts_code=code, adj='qfq', start_date=warmup_start, end_date=end_date, api=pro)
                    if df is not None and not df.empty:
                        # 立即计算信号以节省后续内存/处理时间
                        df_sig = calculate_signals_vectorized(df)
                        # 设置日期索引并排序
                        df_sig['trade_date'] = pd.to_datetime(df_sig['trade_date'])
                        df_sig.set_index('trade_date', inplace=True)
                        df_sig.sort_index(inplace=True)
                        data_map[code] = df_sig
                    time.sleep(0.05)
                    break # Success, exit retry loop
                except Exception as e:
                    print(f"Error fetching {code} (Attempt {attempt+1}/3): {e}")
                    time.sleep(1) # Wait before retry
                
    return data_map

# --- Backtest Engine ---

def run_backtest():
    # 1. Setup & Dynamic Pool Generation (初始化与动态股票池生成)
    print("Generating dynamic stock pools...")
    schedule = get_rebalance_schedule(START_DATE, END_DATE)
    
    pool_map = {} # date -> list of codes (日期 -> 股票代码列表)
    all_involved_stocks = set()
    
    # 找到回测开始日期前的最近一次调仓日，作为初始股票池
    initial_rebalance = None
    for item in reversed(schedule):
        if item['date'] <= START_DATE:
            initial_rebalance = item
            break
    
    # 筛选出相关的调仓计划 (开始日期之后 + 初始那一次)
    relevant_schedule = []
    if initial_rebalance:
        relevant_schedule.append(initial_rebalance)
    
    relevant_schedule += [s for s in schedule if s['date'] > START_DATE and s['date'] <= END_DATE]
    
    # 去重
    seen_dates = set()
    unique_schedule = []
    for s in relevant_schedule:
        if s['date'] not in seen_dates:
            unique_schedule.append(s)
            seen_dates.add(s['date'])
    
    # 获取每个调仓日的股票池
    stock_info_map = {} # code -> {'name': name, 'market': market}
    for item in unique_schedule:
        codes, infos = get_stock_pool(item['date'], item['period'])
        pool_map[item['date']] = codes
        stock_info_map.update(infos)
        all_involved_stocks.update(codes)
        time.sleep(0.5) # 避免触发接口限流
        
    if not all_involved_stocks:
        print("No stocks found in any pool.")
        return

    print(f"Total unique stocks involved: {len(all_involved_stocks)}")
    
    # 2. Fetch Data (获取数据)
    data_map = fetch_all_data(list(all_involved_stocks), START_DATE, END_DATE)
    
    # Fetch Benchmark Data (获取基准数据: 沪深300, 上证指数, 中证1000)
    print("Fetching benchmark data (CSI 300, Shanghai Composite, CSI 1000)...")
    benchmarks = {
        'CSI 300': '000300.SH',
        'Shanghai Composite': '000001.SH',
        'CSI 1000': '000852.SH'
    }
    benchmark_data = {}
    
    for name, ts_code in benchmarks.items():
        try:
            df_bench = pro.index_daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            df_bench['trade_date'] = pd.to_datetime(df_bench['trade_date'])
            df_bench.set_index('trade_date', inplace=True)
            df_bench.sort_index(inplace=True)
            benchmark_data[name] = df_bench
        except Exception as e:
            print(f"Error fetching benchmark {name}: {e}")
    
    # 3. Get Trading Calendar (获取交易日历)
    cal = pro.trade_cal(exchange='', start_date=START_DATE, end_date=END_DATE, is_open='1')
    trade_dates = pd.to_datetime(cal['cal_date']).sort_values().tolist()
    
    if not trade_dates:
        print("No trade dates found.")
        return

    print(f"Backtesting range: {trade_dates[0].date()} to {trade_dates[-1].date()}")
    
    # 4. Initialize State (初始化回测状态)
    cash = INITIAL_CASH
    positions = {} # 持仓字典 {ts_code: {'shares': int, 'buy_date': date, 'buy_price': float}}
    history = [] # 每日资产记录
    daily_stats = [] # 每日详细统计
    trades = [] # 交易记录
    last_sold_info = {} # 记录最近卖出的股票 {code: sell_date}
    total_repo_profit = 0.0 # 累计逆回购收益
    
    total_signals_triggered = 0
    sorted_pool_dates = sorted(pool_map.keys())
    
    print("Starting simulation...")
    
    for current_date in trade_dates:
        current_date_str = current_date.strftime('%Y%m%d')
        
        # Determine active pool (确定当前生效的股票池)
        # Switch to new pool on the day AFTER the rebalance date to avoid look-ahead bias
        # (Pool is calculated using rebalance date's closing data, so we can only trade it next day)
        # 关键修正：将 <= 改为 <，确保在调仓日次日才使用新股票池，避免使用当日收盘数据交易当日开盘
        active_pool_date = sorted_pool_dates[0]
        for d in sorted_pool_dates:
            if d < current_date_str: 
                active_pool_date = d
            else:
                break
        current_stock_list = pool_map.get(active_pool_date, [])

        # --- 0. Exit Logic (Stop Loss & Take Profit) (退出逻辑：止损与止盈) ---
        # Check for stop loss and take profit conditions BEFORE other actions
        # 优先检查止损和止盈条件
        for code in list(positions.keys()):
            pos = positions[code]
            df = data_map.get(code)
            if df is not None and current_date in df.index:
                current_open = df.loc[current_date, 'O']
                current_low = df.loc[current_date, 'L']
                current_high = df.loc[current_date, 'H']
                
                if isinstance(current_open, pd.Series): current_open = current_open.iloc[0]
                if isinstance(current_low, pd.Series): current_low = current_low.iloc[0]
                if isinstance(current_high, pd.Series): current_high = current_high.iloc[0]
                
                stop_price = pos['buy_price'] * (1 + STOP_LOSS_PCT)
                take_profit_price = pos['buy_price'] * (1 + TAKE_PROFIT_PCT)
                
                triggered = False
                sell_price = 0
                reason = ''
                
                # 1. Check Stop Loss (Priority: Risk Control)
                if current_open <= stop_price:
                    # Gapped down below stop loss, sell at Open
                    triggered = True
                    sell_price = current_open
                    reason = 'Stop Loss (Gap Down)'
                elif current_low <= stop_price:
                    # Intraday hit stop loss
                    triggered = True
                    sell_price = stop_price
                    reason = 'Stop Loss (Intraday)'
                
                # 2. Check Take Profit (If not stopped out)
                if not triggered:
                    if current_open >= take_profit_price:
                        # Gapped up above take profit, sell at Open
                        triggered = True
                        sell_price = current_open
                        reason = 'Take Profit (Gap Up)'
                    elif current_high >= take_profit_price:
                        # Intraday hit take profit
                        triggered = True
                        sell_price = take_profit_price
                        reason = 'Take Profit (Intraday)'
                
                if triggered:
                    # Apply Slippage to Sell Price (卖出滑点)
                    # Sell Price decreases by slippage
                    actual_sell_price = sell_price * (1 - SLIPPAGE_RATE)
                    
                    revenue = pos['shares'] * actual_sell_price
                    
                    # Transaction Costs (交易成本)
                    commission = revenue * COMMISSION_RATE
                    stamp_duty = revenue * STAMP_DUTY_RATE
                    total_cost = commission + stamp_duty
                    
                    net_revenue = revenue - total_cost
                    cash += net_revenue
                    
                    # Calculate Profit based on Net Revenue vs Cost (including buy costs)
                    # Note: pos['cost'] should ideally store the total cost including buy commission.
                    # But currently pos['buy_price'] is raw price. 
                    # Let's calculate raw profit for reference, but net profit for account.
                    # To be precise, we should store 'total_buy_cost' in positions.
                    # For now, let's approximate or update buy logic to store cost.
                    # Let's assume we update buy logic later.
                    # Here we calculate profit as (Net Revenue - (Shares * Buy Price)) - Buy Commission?
                    # Better: Net Revenue - Initial Cost (which includes buy commission)
                    
                    # We need to update Buy Logic to store 'total_cost'
                    initial_cost = pos.get('total_cost', pos['shares'] * pos['buy_price'])
                    
                    profit_amount = net_revenue - initial_cost
                    profit_rate = profit_amount / initial_cost
                    
                    trades.append({
                        'date': current_date,
                        'code': code,
                        'name': pos.get('name', code),
                        'action': 'SELL',
                        'price': sell_price, # Log raw price
                        'exec_price': round(actual_sell_price, 2),
                        'shares': pos['shares'],
                        'amount': round(revenue, 2),
                        'commission': round(commission, 2),
                        'tax': round(stamp_duty, 2),
                        'net_amount': round(net_revenue, 2),
                        'pos_pct': f"{pos.get('pos_pct', 0)*100:.1f}%",
                        'profit_amount': round(float(profit_amount), 2),
                        'profit': f"{profit_rate*100:.1f}%",
                        'held_days': (current_date - pos['buy_date']).days,
                        'reason': reason
                    })
                    
                    # Record sell date for cooldown
                    last_sold_info[code] = current_date
                    
                    del positions[code]

        # --- A. Buy Logic (买入逻辑) ---
        # Check for signals on PREVIOUS day (to buy at Open today)
        # 检查前一天的信号（以便在今天开盘买入）
        # Find previous trade date
        try:
            curr_idx = trade_dates.index(current_date)
            if curr_idx > 0:
                prev_date = trade_dates[curr_idx - 1]
                
                # Find candidates (寻找候选股)
                candidates = []
                # Only check stocks in the current active pool (仅检查当前生效股票池中的股票)
                for code in current_stock_list:
                    if code not in positions:
                        df = data_map.get(code)
                        if df is not None and prev_date in df.index:
                            row = df.loc[prev_date]
                            if row['signal']:
                                candidates.append({
                                    'code': code,
                                    'score': row['买入总分']
                                })
                
                total_signals_triggered += len(candidates)
                
                # Sort by score (按分数排序，优先买入高分股)
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # Buy (执行买入)
                for cand in candidates:
                    if len(positions) >= MAX_POSITIONS:
                        break
                    
                    if cand['code'] in positions:
                        continue

                    # Cooldown Check (冷却期检查)
                    if cand['code'] in last_sold_info:
                        last_sell_date = last_sold_info[cand['code']]
                        if (current_date - last_sell_date).days < COOLDOWN_DAYS:
                            continue # Skip if within cooldown period

                    # Calculate current equity for position sizing (Compound Interest) (计算当前权益以进行复利投资)
                    current_market_value = 0
                    for p_code, p_pos in positions.items():
                        p_df = data_map.get(p_code)
                        if p_df is not None and current_date in p_df.index:
                            p_price = p_df.loc[current_date, 'O']
                            if isinstance(p_price, pd.Series): p_price = p_price.iloc[0]
                            current_market_value += p_pos['shares'] * p_price
                        else:
                            current_market_value += p_pos['shares'] * p_pos['buy_price'] # Fallback
                    
                    current_total_equity = cash + current_market_value
                    target_amt = current_total_equity * POSITION_SIZE_PCT

                    if cash < target_amt:
                        # If cash is less than target amount, check if we can buy at least some shares?
                        # Or strictly require target_amt? 
                        # Usually if cash < target_amt, we buy with remaining cash if it's significant?
                        # But let's stick to the logic: if cash < target_amt, we might not be able to buy full position.
                        # Let's try to buy as much as possible up to target_amt, but limited by cash.
                        # Actually, the original logic was: if cash < (INITIAL_CASH * POSITION_SIZE_PCT): break
                        # Now we should check if cash is too small to make a meaningful trade.
                        if cash < (current_total_equity * 0.01): # Less than 1% of equity
                             break 
                    
                    # Use min(target_amt, cash) to determine actual buy amount
                    buy_amt = min(target_amt, cash)
                        
                    code = cand['code']
                    df = data_map.get(code)
                    if df is not None and current_date in df.index:
                        # Check for Limit Up/Down (涨跌停检查)
                        # If High == Low, it's a one-bar limit move (likely un-tradeable)
                        # 如果全天最高价等于最低价，说明是一字板，无法交易
                        day_high = df.loc[current_date, 'H']
                        day_low = df.loc[current_date, 'L']
                        if isinstance(day_high, pd.Series): day_high = day_high.iloc[0]
                        if isinstance(day_low, pd.Series): day_low = day_low.iloc[0]
                        
                        if day_high == day_low:
                            continue # Skip trade (Limit Up/Down)

                        buy_price = df.loc[current_date, 'O']
                        # Handle potential duplicate index or Series return (处理可能的重复索引或Series返回)
                        if isinstance(buy_price, pd.Series):
                            buy_price = buy_price.iloc[0]
                            
                        if np.isnan(buy_price): continue
                        
                        # Check if Open is Limit Up (Approximate check: > 9.5% from prev close)
                        # 检查开盘是否涨停 (近似检查: 较昨收涨幅 > 9.5%)
                        # We need prev close.
                        prev_close = df.loc[current_date, 'pre_close'] if 'pre_close' in df.columns else None
                        if prev_close is None:
                             # Try to get from prev row
                             idx_loc = df.index.get_loc(current_date)
                             if idx_loc > 0:
                                 prev_close = df.iloc[idx_loc-1]['C']
                        
                        if prev_close is not None:
                            if isinstance(prev_close, pd.Series): prev_close = prev_close.iloc[0]
                            if prev_close > 0:
                                open_pct = (buy_price - prev_close) / prev_close
                                if open_pct > 0.095: # Limit Up at Open
                                    continue

                        # Calculate shares (round to 100) (计算股数，向下取整到100股)
                        # Adjust buy_price for slippage to check affordability
                        est_buy_price = buy_price * (1 + SLIPPAGE_RATE)
                        shares = int(buy_amt / est_buy_price / 100) * 100
                        
                        if shares > 0:
                            # Calculate actual costs
                            actual_buy_price = buy_price * (1 + SLIPPAGE_RATE)
                            raw_cost = shares * actual_buy_price
                            commission = raw_cost * COMMISSION_RATE
                            total_cost = raw_cost + commission
                            
                            if cash >= total_cost:
                                cash -= total_cost
                                
                                # Estimate equity for logging
                                pos_pct = total_cost / current_total_equity if current_total_equity > 0 else 0
                                
                                stock_info = stock_info_map.get(code, {'name': code, 'market': 'Unknown'})
                                positions[code] = {
                                    'shares': shares,
                                    'buy_date': current_date,
                                    'buy_price': buy_price, # Store raw price for reference
                                    'total_cost': total_cost, # Store total cost for profit calc
                                    'name': stock_info['name'],
                                    'market': stock_info['market'],
                                    'pos_pct': pos_pct
                                }
                                trades.append({
                                    'date': current_date,
                                    'code': code,
                                    'name': stock_info['name'],
                                    'market': stock_info['market'],
                                    'action': 'BUY',
                                    'price': buy_price,
                                    'exec_price': round(actual_buy_price, 2),
                                    'shares': shares,
                                    'amount': round(raw_cost, 2),
                                    'commission': round(commission, 2),
                                    'total_cost': round(total_cost, 2),
                                    'pos_pct': f"{pos_pct*100:.1f}%",
                                    'score': cand['score']
                                })
                            else:
                                # Insufficient cash (due to commission/rounding), skip or adjust
                                # For now, just skip to avoid "Free Stock" bug
                                continue

        except ValueError:
            pass

        # --- B. Sell Logic (卖出逻辑) ---
        # Check holding period (检查持仓周期)
        # We iterate a copy of keys to allow modification (遍历副本以允许修改字典)
        for code in list(positions.keys()):
            pos = positions[code]
            held_days = 0
            
            # Calculate trading days held (计算持仓交易天数)
            # Simple approach: count days since buy_date in trade_dates list
            # Find index of buy_date
            try:
                buy_idx = trade_dates.index(pos['buy_date'])
                curr_idx = trade_dates.index(current_date)
                held_days = curr_idx - buy_idx
            except ValueError:
                held_days = 0 # Should not happen
            
            if held_days >= HOLD_DAYS:
                # Sell at Close of this day (or Open? Let's use Close to be safe/conservative or Open of next? 
                # User said "hold for 25 days". 
                # Let's sell at today's OPEN if we held for 25 days already? 
                # Or sell at today's CLOSE if today is the 25th day?
                # Let's assume sell at today's OPEN if held_days >= 25.
                # 达到持仓天数，在当日开盘卖出
                
                # Get price
                df = data_map.get(code)
                if df is not None and current_date in df.index:
                    # Check for Limit Up/Down (涨跌停检查)
                    day_high = df.loc[current_date, 'H']
                    day_low = df.loc[current_date, 'L']
                    if isinstance(day_high, pd.Series): day_high = day_high.iloc[0]
                    if isinstance(day_low, pd.Series): day_low = day_low.iloc[0]
                    
                    if day_high == day_low:
                        continue # Cannot sell (Limit Down likely)

                    # Sell at Open (以开盘价卖出)
                    sell_price = df.loc[current_date, 'O']
                    # Handle potential duplicate index or Series return
                    if isinstance(sell_price, pd.Series):
                        sell_price = sell_price.iloc[0]
                    
                    # Check if Open is Limit Down (Approximate check: < -9.5% from prev close)
                    # 检查开盘是否跌停
                    prev_close = df.loc[current_date, 'pre_close'] if 'pre_close' in df.columns else None
                    if prev_close is None:
                            idx_loc = df.index.get_loc(current_date)
                            if idx_loc > 0:
                                prev_close = df.iloc[idx_loc-1]['C']
                    
                    if prev_close is not None:
                        if isinstance(prev_close, pd.Series): prev_close = prev_close.iloc[0]
                        if prev_close > 0:
                            open_pct = (sell_price - prev_close) / prev_close
                            if open_pct < -0.095: # Limit Down at Open
                                continue # Cannot sell

                    # Apply Slippage
                    actual_sell_price = sell_price * (1 - SLIPPAGE_RATE)
                    revenue = pos['shares'] * actual_sell_price
                    
                    # Transaction Costs
                    commission = revenue * COMMISSION_RATE
                    stamp_duty = revenue * STAMP_DUTY_RATE
                    total_cost = commission + stamp_duty
                    
                    net_revenue = revenue - total_cost
                    cash += net_revenue
                    
                    initial_cost = pos.get('total_cost', pos['shares'] * pos['buy_price'])
                    profit_amount = net_revenue - initial_cost
                    profit_rate = profit_amount / initial_cost
                    
                    trades.append({
                        'date': current_date,
                        'code': code,
                        'name': pos.get('name', code),
                        'market': pos.get('market', 'Unknown'),
                        'action': 'SELL',
                        'price': sell_price,
                        'exec_price': round(actual_sell_price, 2),
                        'shares': pos['shares'],
                        'amount': round(revenue, 2),
                        'commission': round(commission, 2),
                        'tax': round(stamp_duty, 2),
                        'net_amount': round(net_revenue, 2),
                        'pos_pct': f"{pos.get('pos_pct', 0)*100:.1f}%",
                        'profit_amount': round(float(profit_amount), 2),
                        'profit': f"{profit_rate*100:.1f}%",
                        'held_days': held_days,
                        'reason': reason
                    })
                    
                    # Record sell date for cooldown
                    last_sold_info[code] = current_date
                    
                    del positions[code]

        # --- D. Reverse Repo (国债逆回购) ---
        # Calculate interest on idle cash (conservative 1.4% annualized)
        # Only on trading days, assuming 1 day interest per trading day
        # 仅对剩余资金进行操作，不影响次日交易
        if cash > 1000: # Minimum threshold (最小起投金额)
            daily_repo_interest = cash * REPO_RATE / 365
            cash += daily_repo_interest
            total_repo_profit += daily_repo_interest

        # --- C. Update Equity (更新账户净值) ---
        market_value = 0
        for code, pos in positions.items():
            df = data_map.get(code)
            if df is not None:
                # Use Close price if available, else last known (使用收盘价计算市值)
                if current_date in df.index:
                    price = df.loc[current_date, 'C']
                else:
                    # Fallback to buy price or previous close (simplified)
                    price = pos['buy_price'] 
                market_value += pos['shares'] * price
        
        total_equity = cash + market_value
        history.append({
            'date': current_date,
            'equity': total_equity,
            'cash': cash,
            'positions': len(positions)
        })
        
        daily_stats.append({
            'date': current_date,
            'total_equity': total_equity,
            'cash': cash,
            'market_value': market_value,
            'positions_count': len(positions),
            'repo_profit': total_repo_profit
        })

    # --- 5. Analysis & Plotting (分析与绘图) ---
    # Close all open positions for reporting purposes (to match Chart Equity)
    # 强制平仓所有剩余持仓以便统计（按最后一日收盘价计算浮盈）
    for code, pos in positions.items():
        # Get last price
        df = data_map.get(code)
        last_price = pos['buy_price']
        if df is not None and not df.empty:
             # Try to get price on last day, or last available
             if trade_dates[-1] in df.index:
                 last_price = df.loc[trade_dates[-1], 'C']
             else:
                 last_price = df.iloc[-1]['C']
        
        # Apply Slippage & Costs for hypothetical liquidation
        actual_sell_price = last_price * (1 - SLIPPAGE_RATE)
        revenue = pos['shares'] * actual_sell_price
        commission = revenue * COMMISSION_RATE
        stamp_duty = revenue * STAMP_DUTY_RATE
        total_cost = commission + stamp_duty
        net_revenue = revenue - total_cost
        
        initial_cost = pos.get('total_cost', pos['shares'] * pos['buy_price'])
        profit_amount = net_revenue - initial_cost
        profit_rate = profit_amount / initial_cost
        
        trades.append({
            'date': trade_dates[-1],
            'code': code,
            'name': pos.get('name', code),
            'market': pos.get('market', 'Unknown'),
            'action': 'HELD_END',
            'price': last_price,
            'exec_price': round(actual_sell_price, 2),
            'shares': pos['shares'],
            'amount': round(revenue, 2),
            'commission': round(commission, 2),
            'tax': round(stamp_duty, 2),
            'net_amount': round(net_revenue, 2),
            'pos_pct': f"{pos.get('pos_pct', 0)*100:.1f}%",
            'profit_amount': round(float(profit_amount), 2),
            'profit': f"{profit_rate*100:.1f}%",
            'held_days': (trade_dates[-1] - pos['buy_date']).days,
            'reason': 'End of Backtest'
        })

    df_res = pd.DataFrame(history)
    df_res.set_index('date', inplace=True)
    
    # Calculate Daily Returns (计算日收益率)
    df_res['pct_chg'] = df_res['equity'].pct_change().fillna(0)
    
    # --- Advanced Metrics (高级指标计算) ---
    
    # 1. Basic Return Metrics (基础收益指标)
    final_equity = df_res.iloc[-1]['equity']
    total_return = (final_equity - INITIAL_CASH) / INITIAL_CASH
    days = (df_res.index[-1] - df_res.index[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    # 2. Trade Statistics (交易统计)
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        # Need to parse profit back to float for stats calculation
        # Filter only SELL and HELD_END for stats? Or just SELL?
        # Usually stats are for closed trades.
        completed_trades = df_trades[df_trades['action'] == 'SELL'].copy()
        if not completed_trades.empty:
            completed_trades['profit_float'] = completed_trades['profit'].str.rstrip('%').astype(float) / 100
            
            total_trades = len(completed_trades)
            winning_trades = len(completed_trades[completed_trades['profit_float'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_holding_period = completed_trades['held_days'].mean()
            
            avg_win = completed_trades[completed_trades['profit_float'] > 0]['profit_float'].mean()
            avg_loss = completed_trades[completed_trades['profit_float'] <= 0]['profit_float'].mean()
            profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            trading_frequency = total_trades / (days / 30) if days > 0 else 0 # Trades per month
        else:
            total_trades = 0
            win_rate = 0
            avg_holding_period = 0
            profit_loss_ratio = 0
            trading_frequency = 0
    else:
        total_trades = 0
        win_rate = 0
        avg_holding_period = 0
        profit_loss_ratio = 0
        trading_frequency = 0

    # 3. Risk Metrics (风险指标)
    # Risk Free Rate (无风险利率)
    rf = RISK_FREE_RATE
    daily_rf = (1 + rf) ** (1/252) - 1
    
    excess_daily_ret = df_res['pct_chg'] - daily_rf
    std_dev = df_res['pct_chg'].std()
    annualized_volatility = std_dev * np.sqrt(252)
    
    # Sharpe Ratio (夏普比率)
    sharpe = (excess_daily_ret.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
    
    # Sortino Ratio (Downside Deviation) (索提诺比率 - 下行偏差)
    downside_returns = df_res['pct_chg'][df_res['pct_chg'] < 0]
    downside_std = downside_returns.std()
    sortino = (excess_daily_ret.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    
    # Max Drawdown Calculation (最大回撤计算)
    df_res['peak'] = df_res['equity'].cummax()
    df_res['dd'] = (df_res['equity'] - df_res['peak']) / df_res['peak']
    max_dd = df_res['dd'].min()
    
    # Find Max Drawdown Interval (Peak to Trough) (寻找最大回撤区间 - 峰值到谷底)
    max_dd_end_date = df_res['dd'].idxmin()
    peak_val_at_dd = df_res.loc[max_dd_end_date, 'peak']
    # Find the last date before max_dd_end_date where equity was >= peak_val_at_dd
    # This is the start of the drawdown (这是回撤的开始日期)
    temp_df = df_res.loc[:max_dd_end_date]
    # Use a small tolerance for float comparison or just find where it equals peak (使用小容差进行浮点比较或直接查找等于峰值的位置)
    max_dd_start_date = temp_df[temp_df['equity'] >= peak_val_at_dd].index[-1]
    
    # VaR / CVaR (Historical Method, 95% Confidence) (风险价值 / 条件风险价值 - 历史法, 95%置信度)
    var_95 = np.percentile(df_res['pct_chg'], 5)
    cvar_95 = df_res['pct_chg'][df_res['pct_chg'] <= var_95].mean()
    
    # Distribution Metrics (分布指标)
    skewness = df_res['pct_chg'].skew()
    kurtosis = df_res['pct_chg'].kurt()

    # --- Print Results (打印结果) ---
    print("\n" + "="*40)
    print("          BACKTEST RESULTS          ")
    print("="*40)
    print(f"Time Range:      {df_res.index[0].date()} to {df_res.index[-1].date()} ({days} days)")
    print(f"Initial Capital: {INITIAL_CASH:,.2f}")
    print(f"Final Equity:    {final_equity:,.2f}")
    print("-" * 40)
    print(f"Total Return:    {total_return*100:.2f}%")
    print(f"Annualized Ret:  {annualized_return*100:.2f}%")
    print(f"Max Drawdown:    {max_dd*100:.2f}%")
    print("-" * 40)
    print(f"Stock Profit:    {(final_equity - INITIAL_CASH - total_repo_profit):,.2f}")
    print(f"Repo Profit:     {total_repo_profit:,.2f} (Risk-free)")
    print("-" * 40)
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Sortino Ratio:   {sortino:.2f}")
    print("-" * 40)
    print(f"Win Rate:        {win_rate*100:.2f}%")
    print(f"Profit/Loss Ratio: {profit_loss_ratio:.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Avg Holding Days: {avg_holding_period:.1f}")
    print(f"Trades/Month:    {trading_frequency:.1f}")
    print("-" * 40)
    print(f"Volatility (Ann): {annualized_volatility*100:.2f}%")
    print(f"Skewness:        {skewness:.2f}")
    print(f"Kurtosis:        {kurtosis:.2f}")
    print(f"VaR (95%):       {var_95*100:.2f}%")
    print(f"CVaR (95%):      {cvar_95*100:.2f}%")
    print("="*40)

    # --- Plotting (绘图) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. Strategy Equity (Normalized to 1) (策略净值 - 归一化为1)
    df_res['norm_equity'] = df_res['equity'] / INITIAL_CASH
    ax1.plot(df_res.index, df_res['norm_equity'], label='Strategy', linewidth=2, color='#1f77b4')
    
    # 2. Benchmarks (Normalized to 1) (基准指数 - 归一化为1)
    colors = {'CSI 300': '#ff7f0e', 'Shanghai Composite': '#2ca02c', 'CSI 1000': '#d62728'}
    csi300_norm = None
    
    for name, df_bench in benchmark_data.items():
        # Align dates (对齐日期)
        common_dates = df_res.index.intersection(df_bench.index)
        if not common_dates.empty:
            # Reindex and fill forward (重置索引并前向填充)
            bench_series = df_bench.loc[common_dates, 'close']
            # Normalize to start at 1 (归一化为1)
            bench_norm = bench_series / bench_series.iloc[0]
            ax1.plot(common_dates, bench_norm, label=name, linestyle='--', alpha=0.7, linewidth=1.5, color=colors.get(name, 'gray'))
            
            if name == 'CSI 300':
                # Prepare CSI 300 norm for excess return calculation
                # Reindex to match df_res for subtraction, filling missing with NaN or ffill
                csi300_norm = bench_norm.reindex(df_res.index, method='ffill')

    # 3. Highlight Max Drawdown Interval (Peak to Trough) (高亮最大回撤区间 - 峰值到谷底)
    ax1.axvspan(max_dd_start_date, max_dd_end_date, color=CHART_DRAWDOWN_COLOR, alpha=CHART_DRAWDOWN_ALPHA, label=f'Max Drawdown {max_dd*100:.2f}%')

    # Title with more metrics (标题包含更多指标)
    title_str = (
        f"Backtest Performance\n"
        f"Ann. Ret: {annualized_return*100:.1f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | MaxDD: {max_dd*100:.1f}%\n"
        f"Win Rate: {win_rate*100:.1f}% | P/L Ratio: {profit_loss_ratio:.2f} | Trades/Mo: {trading_frequency:.1f}\n"
        f"VaR(95%): {var_95*100:.2f}% | Skew: {skewness:.2f}"
    )
    ax1.set_title(title_str, fontsize=CHART_FONT_SIZE_TITLE)
    
    ax1.set_ylabel('Normalized Equity (Base=1.0)', fontsize=CHART_FONT_SIZE_LABEL)
    ax1.tick_params(axis='both', labelsize=CHART_FONT_SIZE_TICK)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=CHART_FONT_SIZE_LEGEND)
    
    # 4. Excess Return Chart (Middle Subplot) (超额收益图 - 中间子图)
    if csi300_norm is not None:
        excess_ret = df_res['norm_equity'] - csi300_norm
        ax2.plot(df_res.index, excess_ret, label='Excess Return vs CSI 300', color='purple', linewidth=1.5)
        ax2.fill_between(df_res.index, excess_ret, 0, where=(excess_ret>=0), facecolor='purple', alpha=0.3)
        ax2.fill_between(df_res.index, excess_ret, 0, where=(excess_ret<0), facecolor='gray', alpha=0.3)
        
        # Calculate Max Drawdown for Excess Return (计算超额收益的最大回撤)
        # We treat excess return curve as an asset curve to find its drawdown
        # Since excess return can be negative, we add a constant (e.g. 1) to make it a price series starting at 1
        # Or simply find the max drop from peak in the absolute value series
        
        # Construct a synthetic equity curve for excess return: 1 + excess_ret
        # This represents the relative performance wealth index
        excess_equity = 1 + excess_ret
        excess_peak = excess_equity.cummax()
        excess_dd = (excess_equity - excess_peak) / excess_peak
        max_excess_dd = excess_dd.min()
        
        max_excess_dd_end_date = excess_dd.idxmin()
        peak_val_at_excess_dd = excess_peak.loc[max_excess_dd_end_date]
        
        # Find start date
        temp_excess_df = pd.DataFrame({'equity': excess_equity, 'peak': excess_peak})
        temp_excess_df = temp_excess_df.loc[:max_excess_dd_end_date]
        max_excess_dd_start_date = temp_excess_df[temp_excess_df['equity'] >= peak_val_at_excess_dd].index[-1]
        
        # Highlight Excess Return Drawdown
        ax2.axvspan(max_excess_dd_start_date, max_excess_dd_end_date, color=CHART_DRAWDOWN_COLOR, alpha=CHART_DRAWDOWN_ALPHA, label=f'Max Excess DD {max_excess_dd*100:.2f}%')
        
        # Annotate
        mid_date = max_excess_dd_start_date + (max_excess_dd_end_date - max_excess_dd_start_date) / 2
        ax2.text(mid_date, excess_ret.loc[max_excess_dd_end_date], f"Max DD: {max_excess_dd*100:.2f}%", 
                 color='red', fontsize=10, ha='center', va='top', fontweight='bold')

        ax2.set_ylabel('Excess Return', fontsize=CHART_FONT_SIZE_LABEL)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', fontsize=CHART_FONT_SIZE_LEGEND)
    else:
        ax2.text(0.5, 0.5, 'CSI 300 Data Not Available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    # 5. Net Asset Area Chart (Bottom Subplot) (净资产面积图 - 底部子图)
    ax3.plot(df_res.index, df_res['equity'], color='#1f77b4', linewidth=1.5)
    ax3.fill_between(df_res.index, df_res['equity'], 0, color='lightblue', alpha=0.4)
    
    # Annotate Min/Max Equity (标注最低/最高资产)
    min_equity = df_res['equity'].min()
    max_equity = df_res['equity'].max()
    min_date = df_res['equity'].idxmin()
    max_date = df_res['equity'].idxmax()
    
    ax3.annotate(f'Min: {min_equity:,.0f}', xy=(min_date, min_equity), 
                 xytext=(0, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='green'),
                 fontsize=10, ha='center', color='green', fontweight='bold')
                 
    ax3.annotate(f'Max: {max_equity:,.0f}', xy=(max_date, max_equity), 
                 xytext=(0, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'),
                 fontsize=10, ha='center', color='red', fontweight='bold')

    ax3.set_ylabel('Net Asset Value', fontsize=CHART_FONT_SIZE_LABEL)
    ax3.set_xlabel('Date', fontsize=CHART_FONT_SIZE_LABEL)
    ax3.tick_params(axis='both', labelsize=CHART_FONT_SIZE_TICK)
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Adjust Y-axis to focus on the data range but keep the "area" feel
    y_min_limit = min(INITIAL_CASH, min_equity) * 0.85
    ax3.set_ylim(bottom=y_min_limit)
    
    # Add text box with key stats (Simplified since title has most) (添加关键统计数据文本框)
    stats_text = (
        f"Total Ret: {total_return*100:.1f}%\n"
        f"Total Trades: {total_trades}"
    )
    ax1.text(0.02, 0.60, stats_text, transform=ax1.transAxes, 
             fontsize=CHART_FONT_SIZE_LEGEND, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('backtest_result.png', dpi=300)
    print("Chart saved to backtest_result.png")
    
    # Save daily stats
    if daily_stats:
        pd.DataFrame(daily_stats).to_csv('backtest_daily.csv', index=False)
        print("Daily stats saved to backtest_daily.csv")
    
    # Save trades (保存交易记录)
    if not df_trades.empty:
        df_trades.to_csv('backtest_trades.csv', index=False)
        print("Trades saved to backtest_trades.csv")
        
        # --- Detailed Analysis (详细分析) ---
        analyze_trades(df_trades)

def analyze_trades(df_trades):
    """
    Analyze trade history and print detailed statistics.
    """
    print("\n" + "="*40)
    print("          DETAILED TRADE ANALYSIS          ")
    print("="*40)
    
    if df_trades.empty:
        print("No trades to analyze.")
        return

    # Filter for closed trades (SELL or HELD_END)
    closed_trades = df_trades[df_trades['action'].isin(['SELL', 'HELD_END'])].copy()
    if closed_trades.empty:
        print("No closed trades.")
        return
        
    closed_trades['profit_float'] = closed_trades['profit'].str.rstrip('%').astype(float) / 100
    closed_trades['year'] = pd.to_datetime(closed_trades['date']).dt.year
    
    # 1. Analysis by Year (按年份分析)
    print("\n--- Performance by Year ---")
    yearly_stats = closed_trades.groupby('year').agg({
        'profit_amount': 'sum',
        'profit_float': 'mean',
        'code': 'count'
    }).rename(columns={'code': 'trades', 'profit_float': 'avg_return'})
    
    # Calculate Win Rate by Year
    yearly_wins = closed_trades[closed_trades['profit_float'] > 0].groupby('year')['code'].count()
    yearly_stats['win_rate'] = (yearly_wins / yearly_stats['trades']).fillna(0)
    
    # Format output
    print(f"{'Year':<6} | {'Trades':<6} | {'Win Rate':<8} | {'Avg Return':<10} | {'Total Profit':<12}")
    print("-" * 55)
    for year, row in yearly_stats.iterrows():
        print(f"{year:<6} | {int(row['trades']):<6} | {row['win_rate']*100:6.1f}% | {row['avg_return']*100:9.2f}% | {row['profit_amount']:12,.2f}")

    # 2. Analysis by Sell Reason (按卖出原因分析)
    print("\n--- Performance by Exit Reason ---")
    reason_stats = closed_trades.groupby('reason').agg({
        'profit_amount': 'sum',
        'profit_float': 'mean',
        'code': 'count'
    }).rename(columns={'code': 'trades', 'profit_float': 'avg_return'})
    
    print(f"{'Reason':<25} | {'Trades':<6} | {'Avg Return':<10} | {'Total Profit':<12}")
    print("-" * 60)
    for reason, row in reason_stats.iterrows():
        print(f"{reason:<25} | {int(row['trades']):<6} | {row['avg_return']*100:9.2f}% | {row['profit_amount']:12,.2f}")

    # 3. Analysis by Market (按板块分析)
    if 'market' in closed_trades.columns:
        print("\n--- Performance by Market ---")
        market_stats = closed_trades.groupby('market').agg({
            'profit_amount': 'sum',
            'profit_float': 'mean',
            'code': 'count'
        }).rename(columns={'code': 'trades', 'profit_float': 'avg_return'})
        
        print(f"{'Market':<10} | {'Trades':<6} | {'Avg Return':<10} | {'Total Profit':<12}")
        print("-" * 50)
        for market, row in market_stats.iterrows():
            print(f"{market:<10} | {int(row['trades']):<6} | {row['avg_return']*100:9.2f}% | {row['profit_amount']:12,.2f}")

    # 4. Top Winners & Losers (最佳/最差个股)
    print("\n--- Top 5 Profitable Trades ---")
    top_winners = closed_trades.nlargest(5, 'profit_float')
    for _, row in top_winners.iterrows():
        print(f"{row['date'].date()} {row['code']} {row['name']} ({row['market']}): {row['profit']} ({row['profit_amount']:,.0f}) [{row['reason']}]")
        
    print("\n--- Top 5 Loss Trades ---")
    top_losers = closed_trades.nsmallest(5, 'profit_float')
    for _, row in top_losers.iterrows():
        print(f"{row['date'].date()} {row['code']} {row['name']} ({row['market']}): {row['profit']} ({row['profit_amount']:,.0f}) [{row['reason']}]")

if __name__ == "__main__":
    run_backtest()