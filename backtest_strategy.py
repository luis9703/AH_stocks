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
# Tushare Token配置，优先从环境变量获取，否则使用默认值
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "ac85143eaf1a537517703687c0596b2a303696345e0162612af7ca9d")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Backtest Parameters (回测参数)
# 回测时间范围：过去3年到今天
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime('%Y%m%d')
END_DATE = datetime.datetime.now().strftime('%Y%m%d')
INITIAL_CASH = 2000000       # 初始资金 200万
MAX_POSITIONS = 20           # 最大持仓股票数量
POSITION_SIZE_PCT = 0.05     # 单只股票仓位占比 5%
HOLD_DAYS = 40               # 持仓天数

# Stock Filter Parameters (选股过滤参数)
FILTER_MIN_MARKET_CAP = 400 * 10000 * 10000  # 最小市值 400亿
FILTER_MIN_PE = 10           # 最小市盈率(TTM)
FILTER_MAX_PE = 60           # 最大市盈率(TTM)
FILTER_MIN_PROFIT_YOY = -25  # 净利润同比增长率下限 (%)
# FILTER_FINANCIAL_PERIOD removed - will be dynamic (财务周期动态获取)

# Strategy Parameters (策略参数)
# 买入评分权重：主力吸货(40%) + 风险控制(35%) + 动量涨跌(25%)
BUY_WEIGHTS = {'s1_main': 0.40, 's2_risk': 0.35, 's3_momentum': 0.25}
BUY_THRESHOLD = 80           # 买入总分阈值
STOP_LOSS_PCT = -0.04        # 止损百分比 (-4%)
TAKE_PROFIT_PCT = 0.16       # 止盈百分比 (16%)
COOLDOWN_DAYS = 5           # 卖出后冷却天数 (禁止买入刚卖出的股票)

# Chart Parameters (图表参数)
CHART_FONT_SIZE_TITLE = 20
CHART_FONT_SIZE_LABEL = 15
CHART_FONT_SIZE_TICK = 12
CHART_FONT_SIZE_LEGEND = 12
CHART_DRAWDOWN_COLOR = 'red'
CHART_DRAWDOWN_ALPHA = 0.3

# Risk Metrics Parameters (风险指标参数)
RISK_FREE_RATE = 0.02        # 无风险利率 2%

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
        cal = pro.trade_cal(exchange='', start_date='20100101', end_date=target_date, is_open='1')
        if cal.empty:
            return [], {}
        last_trade_date = cal.iloc[-1]['cal_date']
        
        # 2. 获取该日期的基础指标 (市值, PE)
        df_daily = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,pe_ttm,total_mv')
        if df_daily.empty:
             print(f"Warning: No daily basic data for {last_trade_date}")
             return [], {}

        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # 过滤: 市值 & PE
        subset = df[
            (df['total_mv'] * 10000 >= FILTER_MIN_MARKET_CAP) & 
            (df['pe_ttm'] >= FILTER_MIN_PE) & 
            (df['pe_ttm'] <= FILTER_MAX_PE)
        ].copy()
        
        # 3. 财务过滤 (净利润增长率 > 0) 针对特定财报周期
        codes = subset['ts_code'].tolist()
        if not codes:
            return [], {}
            
        # 批量获取财务指标
        df_fina = pro.fina_indicator(ts_code=",".join(codes), period=period, fields='ts_code,dt_netprofit_yoy')
        
        if not df_fina.empty:
            subset = pd.merge(subset, df_fina, on='ts_code', how='inner')
            subset = subset[subset['dt_netprofit_yoy'] > FILTER_MIN_PROFIT_YOY]
            
        # 去重
        subset.drop_duplicates(subset=['ts_code'], inplace=True)

        print(f"  > Pool size: {len(subset)}")
        return subset['ts_code'].tolist(), dict(zip(subset['ts_code'], subset['name']))
        
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
            try:
                df = ts.pro_bar(ts_code=code, adj='qfq', start_date=warmup_start, end_date=end_date)
                if df is not None and not df.empty:
                    # 立即计算信号以节省后续内存/处理时间
                    df_sig = calculate_signals_vectorized(df)
                    # 设置日期索引并排序
                    df_sig['trade_date'] = pd.to_datetime(df_sig['trade_date'])
                    df_sig.set_index('trade_date', inplace=True)
                    df_sig.sort_index(inplace=True)
                    data_map[code] = df_sig
                time.sleep(0.05)
            except Exception as e:
                print(f"Error fetching {code}: {e}")
                
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
    stock_name_map = {}
    for item in unique_schedule:
        codes, names = get_stock_pool(item['date'], item['period'])
        pool_map[item['date']] = codes
        stock_name_map.update(names)
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
    trades = [] # 交易记录
    last_sold_info = {} # 记录最近卖出的股票 {code: sell_date}
    
    total_signals_triggered = 0
    sorted_pool_dates = sorted(pool_map.keys())
    
    print("Starting simulation...")
    
    for current_date in trade_dates:
        current_date_str = current_date.strftime('%Y%m%d')
        
        # Determine active pool (确定当前生效的股票池)
        active_pool_date = sorted_pool_dates[0]
        for d in sorted_pool_dates:
            if d <= current_date_str:
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
                    revenue = pos['shares'] * sell_price
                    cash += revenue
                    
                    profit_rate = (sell_price - pos['buy_price']) / pos['buy_price']
                    profit_amount = revenue - (pos['shares'] * pos['buy_price'])
                    
                    trades.append({
                        'date': current_date,
                        'code': code,
                        'name': pos.get('name', code),
                        'action': 'SELL',
                        'price': sell_price,
                        'shares': pos['shares'],
                        'amount': revenue,
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
                        buy_price = df.loc[current_date, 'O']
                        # Handle potential duplicate index or Series return (处理可能的重复索引或Series返回)
                        if isinstance(buy_price, pd.Series):
                            buy_price = buy_price.iloc[0]
                            
                        if np.isnan(buy_price): continue
                        
                        # Calculate shares (round to 100) (计算股数，向下取整到100股)
                        shares = int(buy_amt / buy_price / 100) * 100
                        
                        if shares > 0 and cash >= shares * buy_price:
                            cost = shares * buy_price
                            cash -= cost
                            
                            # Estimate equity for logging
                            pos_pct = cost / current_total_equity if current_total_equity > 0 else 0
                            
                            positions[code] = {
                                'shares': shares,
                                'buy_date': current_date,
                                'buy_price': buy_price,
                                'name': stock_name_map.get(code, code),
                                'pos_pct': pos_pct
                            }
                            trades.append({
                                'date': current_date,
                                'code': code,
                                'name': stock_name_map.get(code, code),
                                'action': 'BUY',
                                'price': buy_price,
                                'shares': shares,
                                'amount': cost,
                                'pos_pct': f"{pos_pct*100:.1f}%",
                                'score': cand['score']
                            })

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
                    # Sell at Open (以开盘价卖出)
                    sell_price = df.loc[current_date, 'O']
                    # Handle potential duplicate index or Series return
                    if isinstance(sell_price, pd.Series):
                        sell_price = sell_price.iloc[0]
                    
                    revenue = pos['shares'] * sell_price
                    cash += revenue
                    
                    profit_rate = (sell_price - pos['buy_price']) / pos['buy_price']
                    profit_amount = revenue - (pos['shares'] * pos['buy_price'])
                    
                    trades.append({
                        'date': current_date,
                        'code': code,
                        'name': pos.get('name', code),
                        'action': 'SELL',
                        'price': sell_price,
                        'shares': pos['shares'],
                        'amount': revenue,
                        'pos_pct': f"{pos.get('pos_pct', 0)*100:.1f}%",
                        'profit_amount': round(float(profit_amount), 2),
                        'profit': f"{profit_rate*100:.1f}%",
                        'held_days': held_days,
                        'reason': f'Held {held_days} days'
                    })
                    
                    # Record sell date for cooldown
                    last_sold_info[code] = current_date
                    
                    del positions[code]

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
        
        profit_rate = (last_price - pos['buy_price']) / pos['buy_price']
        market_val = pos['shares'] * last_price
        profit_amount = market_val - (pos['shares'] * pos['buy_price'])
        
        trades.append({
            'date': trade_dates[-1],
            'code': code,
            'name': pos.get('name', code),
            'action': 'HELD_END',
            'price': last_price,
            'shares': pos['shares'],
            'amount': market_val,
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Strategy Equity (Normalized to 1) (策略净值 - 归一化为1)
    df_res['norm_equity'] = df_res['equity'] / INITIAL_CASH
    ax1.plot(df_res.index, df_res['norm_equity'], label='Strategy', linewidth=2, color='#1f77b4')
    
    # 2. Benchmarks (Normalized to 1) (基准指数 - 归一化为1)
    colors = {'CSI 300': '#ff7f0e', 'Shanghai Composite': '#2ca02c', 'CSI 1000': '#d62728'}
    for name, df_bench in benchmark_data.items():
        # Align dates (对齐日期)
        common_dates = df_res.index.intersection(df_bench.index)
        if not common_dates.empty:
            # Reindex and fill forward (重置索引并前向填充)
            bench_series = df_bench.loc[common_dates, 'close']
            # Normalize to start at 1 (归一化为1)
            bench_norm = bench_series / bench_series.iloc[0]
            ax1.plot(common_dates, bench_norm, label=name, linestyle='--', alpha=0.7, linewidth=1.5, color=colors.get(name, 'gray'))

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
    
    # 4. Net Asset Bar Chart (Bottom Subplot) (净资产柱状图 - 底部子图)
    ax2.bar(df_res.index, df_res['equity'], color='#1f77b4', alpha=0.6, width=1.0)
    ax2.set_ylabel('Net Asset Value', fontsize=CHART_FONT_SIZE_LABEL)
    ax2.set_xlabel('Date', fontsize=CHART_FONT_SIZE_LABEL)
    ax2.tick_params(axis='both', labelsize=CHART_FONT_SIZE_TICK)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
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
    
    # Save trades (保存交易记录)
    if not df_trades.empty:
        df_trades.to_csv('backtest_trades.csv', index=False)
        print("Trades saved to backtest_trades.csv")

if __name__ == "__main__":
    run_backtest()