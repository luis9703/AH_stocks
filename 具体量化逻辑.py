import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. 初始化Tushare API ---
try:
    from tushare_token import TOKEN
    ts.set_token(TOKEN)
except ImportError:
    token = "ac85143eaf1a537517703687c0596b2a303696345e0162612af7ca9d" 
    ts.set_token(token)
pro = ts.pro_api()

# 设置Matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def screen_stocks():
    """
    第一步：根据基本面指标筛选股票
    """
    print("--- 步骤 1: 正在执行基本面选股 ---")
    
    # --- 定义筛选条件 ---
    MAX_PE_TTM = 30; MIN_ROE = 10; MIN_MARKET_CAP = 100; MAX_PEG = 1
    print(f"筛选条件: PE < {MAX_PE_TTM}, ROE > {MIN_ROE}%, 市值 > {MIN_MARKET_CAP}亿, 0 < PEG < {MAX_PEG}")

    # 获取所有A股列表
    stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
    stock_list = stock_list[~stock_list['name'].str.contains('ST') & ~stock_list['ts_code'].str.endswith('BJ') & ~stock_list['industry'].isin(['银行', '保险', '证券'])]
    
    # 获取最新交易日
    today = datetime.now()
    trade_cal = pro.trade_cal(exchange='', start_date=(today - timedelta(days=10)).strftime('%Y%m%d'), end_date=today.strftime('%Y%m%d'))
    last_trade_day = trade_cal[trade_cal['is_open'] == 1].cal_date.max()
    
    # 获取所有股票的最新估值指标
    df_daily_basic = pro.daily_basic(trade_date=last_trade_day, fields='ts_code,total_mv,pe_ttm')
    df_daily_basic['total_mv'] = df_daily_basic['total_mv'] / 10000

    # --- 修复后的逻辑 ---
    # 1. 先合并基础信息和估值信息
    df_merged = pd.merge(stock_list, df_daily_basic, on='ts_code', how='inner')

    # 2. 进行第一次筛选（仅基于PE和市值）
    pre_selected_df = df_merged[
        (df_merged['pe_ttm'] > 0) &
        (df_merged['pe_ttm'] < MAX_PE_TTM) &
        (df_merged['total_mv'] > MIN_MARKET_CAP)
    ].copy()

    if pre_selected_df.empty:
        print("基于PE和市值的初选没有找到任何股票。")
        return None

    # 3. 获取筛选后股票列表，一次性查询财务指标 (ROE 和 净利润同比增长率)
    codes_to_query = ",".join(pre_selected_df['ts_code'].tolist())
    
    # 动态获取最新的年报报告期
    latest_report_period = f"{today.year - 1}1231"
    print(f"正在为 {len(pre_selected_df)} 支初选股票获取 {latest_report_period} 的财务数据...")
    
    # 同时获取 roe 和 netprofit_yoy
    df_fina = pro.fina_indicator(ts_code=codes_to_query, period=latest_report_period, fields='ts_code,roe,netprofit_yoy')
    df_fina.dropna(inplace=True)

    # 4. 合并财务数据
    final_df = pd.merge(pre_selected_df, df_fina, on='ts_code', how='inner')
    
    if final_df.empty:
        print("合并财务数据后无有效股票。")
        return None

    # 5. 手动计算PEG
    # 确保 netprofit_yoy > 0 以避免除以零或负数
    final_df = final_df[final_df['netprofit_yoy'] > 0].copy()
    final_df['peg'] = final_df['pe_ttm'] / final_df['netprofit_yoy']

    # 6. 进行最终筛选（基于ROE和计算出的PEG）
    selected_df = final_df[
        (final_df['roe'] > MIN_ROE) &
        (final_df['peg'] < MAX_PEG)
    ].copy()
    
    if selected_df.empty:
        print("根据所有条件，未筛选出任何股票。")
        return None
        
    selected_df = selected_df.sort_values(by='total_mv', ascending=False)
    
    # 动态构建重命名和输出列
    rename_dict = {'name': '名称', 'industry': '行业', 'pe_ttm': 'PE(TTM)', 'peg': 'PEG', 'roe': 'ROE(%)', 'total_mv': '市值(亿)'}
    display_columns = ['ts_code', '名称', '行业', 'PE(TTM)', 'PEG', 'ROE(%)', '市值(亿)']
    
    selected_df.rename(columns=rename_dict, inplace=True)

    print("\n筛选结果如下 (按市值排序):")
    print(selected_df[display_columns].to_string(index=False))
    
    return selected_df

def sma_cn(series, n, m):
    return series.ewm(alpha=m/n, adjust=True).mean()

def calculate_ths_indicator(ts_code: str, end_date: str, freq: str = 'D', fetch_years: int = 3, plot_periods: int = 250):
    print(f"\n--- 步骤 2: 正在对 {ts_code} 进行技术分析 ---")
    
    # --- 可调节参数区域 ---
    buy_weights = {'s1_main': 0.40, 's2_risk': 0.35, 's3_momentum': 0.25}
    buy_threshold = 80
    
    sell_weights = {'overheat': 40, 'momentum_loss': 30, 'trend_break': 30}
    sell_threshold = 70
    
    # 止损价 = 买入价 - 3 * ATR
    stop_loss_config = {'atr_period': 14, 'atr_multiplier': 3.0}
    # --- 参数区域结束 ---

    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=fetch_years * 365)).strftime('%Y%m%d')
    
    if freq == 'D': df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    elif freq == 'W': df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
    elif freq == 'M': df = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else: print(f"错误：不支持的K线级别 '{freq}'"); return
    if df.empty: print("错误：未获取到K线数据。"); return
    df = df.iloc[::-1].reset_index(drop=True)
    df.rename(columns={'close': 'C', 'low': 'L', 'high': 'H'}, inplace=True)
    
    # --- 指标计算 ---
    VAR1 = df['L'].shift(1); sma_abs = sma_cn(abs(df['L'] - VAR1), 3, 1); sma_max = sma_cn(np.maximum(df['L'] - VAR1, 0), 3, 1)
    VAR2 = (sma_abs / sma_max) * 100; VAR2.fillna(0, inplace=True)
    VAR3 = (VAR2 * 10).ewm(span=3, adjust=False).mean(); VAR4 = df['L'].rolling(38).min(); VAR5 = VAR3.rolling(38).max()
    VAR6 = np.where(df['L'] == df['L'].rolling(90).min(), 1, 0)
    condition_var7 = df['L'] <= VAR4; val_if_true = (VAR3 + VAR5 * 2) / 2; ema_base = np.where(condition_var7, val_if_true, 0)
    VAR7 = pd.Series(ema_base, index=df.index).ewm(span=3, adjust=False).mean() / 618 * VAR6
    df['主力吸货'] = VAR7
    llv_21 = df['L'].rolling(21).min(); hhv_21 = df['H'].rolling(21).max(); range_21 = hhv_21 - llv_21
    VAR8 = np.where(range_21 > 0, (df['C'] - llv_21) / range_21 * 100, 0)
    VAR9 = sma_cn(pd.Series(VAR8, index=df.index), 13, 8); df['风险'] = np.ceil(sma_cn(VAR9, 13, 8))
    llv_27 = df['L'].rolling(27).min(); hhv_27 = df['H'].rolling(27).max(); range_27 = hhv_27 - llv_27
    k_stoch_27 = np.where(range_27 > 0, (df['C'] - llv_27) / range_27 * 100, 0)
    sma1 = sma_cn(pd.Series(k_stoch_27, index=df.index), 5, 1); sma2 = sma_cn(sma1, 3, 1)
    inner_calc = 3 * sma1 - 2 * sma2; df['涨跌'] = inner_calc.rolling(5).mean()
    
    # --- 买入评分系统 ---
    score1 = np.zeros(len(df))
    base_cond = df['主力吸货'] > 0; cont_cond = (df['主力吸货'] > df['主力吸货'].shift(1)) & (df['主力吸货'].shift(1) > df['主力吸货'].shift(2)); strength_cond = df['主力吸货'] > df['主力吸货'].rolling(20).mean()
    score1[base_cond] += 50; score1[cont_cond] += 30; score1[strength_cond] += 20
    df['主力吸货分数'] = score1
    score2 = np.zeros(len(df))
    score2[df['风险'] < 20] = 100; score2[(df['风险'] >= 20) & (df['风险'] < 50)] = 70; score2[(df['风险'] >= 50) & (df['风险'] < 80)] = 30
    df['风险分数'] = score2
    score3 = np.zeros(len(df))
    score3[(df['涨跌'] > 0) & (df['涨跌'] > df['涨跌'].shift(1))] = 100; score3[(df['涨跌'] > 0) & (df['涨跌'] <= df['涨跌'].shift(1))] = 50
    df['涨跌分数'] = score3
    df['买入总分'] = df['主力吸货分数'] * buy_weights['s1_main'] + df['风险分数'] * buy_weights['s2_risk'] + df['涨跌分数'] * buy_weights['s3_momentum']
    
    # --- 卖出评分系统 (指标部分) ---
    # 均线以35日为准
    ma35 = df['C'].rolling(35).mean()
    sell_score1 = np.zeros(len(df)); sell_score1[df['风险'] >= 85] = sell_weights['overheat']; sell_score1[(df['风险'] >= 75) & (df['风险'] < 85)] = sell_weights['overheat'] / 2
    momentum_loss_cond = (df['涨跌'] < df['涨跌'].shift(1)) & (df['涨跌'].shift(1) < df['涨跌'].shift(2))
    sell_score2 = np.where(momentum_loss_cond, sell_weights['momentum_loss'], 0)
    sell_score3 = np.zeros(len(df)); over_extended_cond = df['C'] > (ma35 * 1.15); trend_break_cond = df['C'] < ma35
    sell_score3[over_extended_cond] = sell_weights['trend_break'] / 2; sell_score3[trend_break_cond] = sell_weights['trend_break']
    df['卖出总分'] = sell_score1 + sell_score2 + sell_score3

    # --- 动态止损系统 ---
    high_minus_low = df['H'] - df['L']
    high_minus_close_prev = abs(df['H'] - df['C'].shift(1))
    low_minus_close_prev = abs(df['L'] - df['C'].shift(1))
    tr = pd.concat([high_minus_low, high_minus_close_prev, low_minus_close_prev], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/stop_loss_config['atr_period'], adjust=False).mean()
    
    df['止损线'] = 0.0
    df['信号'] = ''
    in_position = False
    highest_since_buy = 0
    
    for i in range(1, len(df)):
        # 优先判断卖出
        if in_position:
            if df.loc[i, 'L'] < df.loc[i-1, '止损线']:
                df.loc[i, '信号'] = '止损卖出'
                in_position = False
                highest_since_buy = 0
            else:
                highest_since_buy = max(highest_since_buy, df.loc[i, 'H'])
                new_stop_loss = highest_since_buy - stop_loss_config['atr_multiplier'] * df.loc[i, 'atr']
                df.loc[i, '止损线'] = max(df.loc[i-1, '止损线'], new_stop_loss)
        
        # 如果未触发止损，再判断其他信号
        if df.loc[i, '信号'] == '':
            if df.loc[i, '买入总分'] >= buy_threshold and not in_position:
                df.loc[i, '信号'] = '评分买入'
                in_position = True
                highest_since_buy = df.loc[i, 'H']
                df.loc[i, '止损线'] = df.loc[i, 'C'] - stop_loss_config['atr_multiplier'] * df.loc[i, 'atr']
            elif df.loc[i, '卖出总分'] >= sell_threshold and in_position:
                 df.loc[i, '信号'] = '评分卖出'
                 in_position = False
                 highest_since_buy = 0
        
        if not in_position:
            df.loc[i, '止损线'] = np.nan # 不在持仓状态时，不显示止损线

    print("\n最近10个周期的评分和信号:")
    print(df[['trade_date', 'C', '买入总分', '卖出总分', '信号']].tail(10).to_string(index=False))

    # --- 可视化 ---
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df_plot = df.tail(plot_periods)
    buy_signals = df_plot[df_plot['信号'] == '评分买入']
    sell_signals = df_plot[(df_plot['信号'] == '评分卖出') | (df_plot['信号'] == '止损卖出')]
    
    freq_map = {'D': '日线', 'W': '周线', 'M': '月线'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    title = f'{ts_code} {freq_map.get(freq, "")} 量化评分分析'
    fig.suptitle(title, fontsize=16)

    # 子图1: K线和买卖信号
    ax1.set_title('收盘价与买卖关注信号')
    ax1.plot(df_plot['trade_date'], df_plot['C'], label='收盘价', color='gray', zorder=1)
    ax1.plot(df_plot['trade_date'], ma35.tail(plot_periods), label='MA35', color='orange', linestyle='--', alpha=0.7, zorder=2)
    ax1.plot(df_plot['trade_date'], df_plot['止损线'], label='移动止损线', color='blue', linestyle=':', alpha=0.9, zorder=3)
    ax1.scatter(buy_signals['trade_date'], buy_signals['C'], marker='^', color='red', s=120, label='买入点', zorder=5)
    ax1.scatter(sell_signals['trade_date'], sell_signals['C'], marker='v', color='green', s=120, label='卖出点', zorder=5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # 子图2: 买卖总分
    ax2.set_title('综合得分')
    ax2.plot(df_plot['trade_date'], df_plot['买入总分'], label='买入总分', color='purple', alpha=0.8)
    ax2.plot(df_plot['trade_date'], df_plot['卖出总分'], label='卖出总分', color='green', alpha=0.8)
    ax2.axhline(buy_threshold, color='red', linestyle='--', label=f'买入阈值 ({buy_threshold}分)', alpha=0.5)
    ax2.axhline(sell_threshold, color='green', linestyle='--', label=f'卖出阈值 ({sell_threshold}分)', alpha=0.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

def technical_analysis_menu(stock_code):
    """
    处理K线级别选择和调用分析函数的辅助函数
    """
    while True:
        print("\n" + "-"*20 + f" 分析 {stock_code} " + "-"*20)
        print("  1: 日线 (近1年)")
        print("  2: 周线 (近3年)")
        print("  3: 月线 (近10年)")
        print("  b: 返回上一级")
        print("  q: 退出程序")
        
        choice = input("请输入您的选择 (1/2/3/b/q): ").strip().lower()
        
        params = {
            '1': {'freq': 'D', 'fetch_years': 3, 'plot_periods': 250},
            '2': {'freq': 'W', 'fetch_years': 6, 'plot_periods': 156},
            '3': {'freq': 'M', 'fetch_years': 15, 'plot_periods': 120},
        }
        
        if choice in params:
            p = params[choice]
            calculate_ths_indicator(
                ts_code=stock_code,
                end_date=datetime.now().strftime("%Y%m%d"),
                freq=p['freq'],
                fetch_years=p['fetch_years'],
                plot_periods=p['plot_periods']
            )
        elif choice == 'b':
            break
        elif choice == 'q':
            print("程序已退出。")
            exit()
        else:
            print("无效输入。")

# --- 主程序入口 ---
if __name__ == '__main__':
    while True:
        print("\n" + "#"*50)
        print("## 主菜单 ##")
        print("#"*50)
        print("  1: 通过基本面筛选股票，然后进行技术分析")
        print("  2: 输入自定义股票代码，直接进行技术分析")
        print("  q: 退出程序")
        print("#"*50)
        
        main_choice = input("请选择操作 (1/2/q): ").strip().lower()

        if main_choice == '1':
            screened_stocks = screen_stocks()
            if screened_stocks is not None:
                while True:
                    print("\n" + "="*50)
                    stock_code_input = input("从上方列表输入您想分析的股票代码 (例如 603198.SH)，或输入 'b' 返回主菜单: ").strip().upper()
                    
                    if stock_code_input == 'B':
                        break 
                    
                    if stock_code_input not in screened_stocks['ts_code'].values:
                        print("错误：输入的股票代码不在筛选列表中，请重新输入。")
                        continue
                    
                    technical_analysis_menu(stock_code_input)

        elif main_choice == '2':
            while True:
                print("\n" + "="*50)
                stock_code_input = input("请输入您想分析的自定义股票代码 (例如 600519.SH)，或输入 'b' 返回主菜单: ").strip().upper()

                if stock_code_input == 'B':
                    break
                
                if not (len(stock_code_input) == 9 and (stock_code_input.endswith('.SH') or stock_code_input.endswith('.SZ'))):
                    print("错误：股票代码格式不正确，请输入9位代码，并以 .SH 或 .SZ 结尾。")
                    continue
                
                technical_analysis_menu(stock_code_input)

        elif main_choice == 'q':
            print("程序已退出。")
            break
        else:
            print("无效输入，请重新选择。")