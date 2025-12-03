import akshare as ak
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import requests
import os
import time
import warnings

# 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 尝试禁用系统代理，防止网络请求被拦截
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['all_proxy'] = ''

# --- 配置区域 (推荐通过环境变量注入，保证安全) ---
# Tushare Token
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "ac85143eaf1a537517703687c0596b2a303696345e0162612af7ca9d") # 请替换为你的 Tushare Token

# 微信推送 (PushPlus)
# 优先从环境变量获取，如果没有则使用默认值（请替换为您的真实Token）
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "7b4ee9c6a01c42009c59e7b1e193b108")

# 邮件推送配置
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
EMAIL_USER = os.environ.get("EMAIL_USER")       # 发件人邮箱
EMAIL_PASS = os.environ.get("EMAIL_PASS")       # 邮箱授权码
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER") # 收件人邮箱

# 选股硬性指标
MIN_MARKET_CAP = 1000 * 10000 * 10000  # 200亿
MIN_PE = 10
MAX_PE = 30
MAX_DEBT_RATIO = 55 # 负债率 60%

# 技术分析参数
BUY_WEIGHTS = {'s1_main': 0.40, 's2_risk': 0.35, 's3_momentum': 0.25}
BUY_THRESHOLD = 80
# 增加回溯天数：周线计算需要至少90个周期(90周≈630天)，加上节假日和缓冲，建议设置为1500天(约4年)
SCAN_LOOKBACK_DAYS = 1500  
API_CHUNK_SIZE = 50       # 财务数据批量获取大小

# --- 指标计算辅助函数 (源自 具体量化逻辑.py) ---

def sma_cn(series, n, m):
    """
    模拟同花顺 SMA 算法: Y = (M*X + (N-M)*Y')/N
    """
    return series.ewm(alpha=m/n, adjust=True).mean()

def calculate_technical_signals(df):
    """
    计算技术指标和评分 (日线/周线通用)
    返回: (是否触发买入, 买入分数, 信号详情字符串)
    """
    if df is None or len(df) < 90:
        return False, 0, ""

    try:
        # 1. 预处理
        # Tushare pro_bar 返回列: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        df = df.copy()
        # 确保列名匹配逻辑 (C, L, H, O)
        if 'close' in df.columns:
            df.rename(columns={'close': 'C', 'low': 'L', 'high': 'H', 'open': 'O'}, inplace=True)
        
        # 确保按日期升序 (旧->新)
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 2. 指标计算
        # VAR1:=REF(LOW,1)
        VAR1 = df['L'].shift(1)
        
        # SMA(ABS(LOW-VAR1),3,1) / SMA(MAX(LOW-VAR1,0),3,1) * 100
        sma_abs = sma_cn(abs(df['L'] - VAR1), 3, 1)
        sma_max = sma_cn(np.maximum(df['L'] - VAR1, 0), 3, 1)
        VAR2 = (sma_abs / sma_max) * 100
        VAR2.fillna(0, inplace=True)
        
        # VAR3:=EMA(IF(CLOSE*1.2,VAR2*10,VAR2/10),3) -> 简化为 VAR2*10
        VAR3 = (VAR2 * 10).ewm(span=3, adjust=False).mean()
        
        # VAR4:=LLV(LOW,38)
        VAR4 = df['L'].rolling(38).min()
        
        # VAR5:=HHV(VAR3,38)
        VAR5 = VAR3.rolling(38).max()
        
        # VAR6:=IF(LLV(LOW,90),1,0) -> 1 (只要有数据)
        VAR6 = 1
        
        # VAR7:=EMA(IF(LOW<=VAR4,(VAR3+VAR5*2)/2,0),3)/618*VAR6
        condition_var7 = df['L'] <= VAR4
        val_if_true = (VAR3 + VAR5 * 2) / 2
        ema_base = np.where(condition_var7, val_if_true, 0)
        VAR7 = pd.Series(ema_base, index=df.index).ewm(span=3, adjust=False).mean() / 618 * VAR6
        df['主力吸货'] = VAR7
        
        # VAR8:=((C-LLV(L,21))/(HHV(H,21)-LLV(L,21)))*100
        llv_21 = df['L'].rolling(21).min()
        hhv_21 = df['H'].rolling(21).max()
        range_21 = hhv_21 - llv_21
        VAR8 = np.where(range_21 > 0, (df['C'] - llv_21) / range_21 * 100, 0)
        
        # VAR9:=SMA(VAR8,13,8)
        VAR9 = sma_cn(pd.Series(VAR8, index=df.index), 13, 8)
        df['风险'] = np.ceil(sma_cn(VAR9, 13, 8))
        
        # 涨跌 (类似 KDJ J值)
        llv_27 = df['L'].rolling(27).min()
        hhv_27 = df['H'].rolling(27).max()
        range_27 = hhv_27 - llv_27
        k_stoch_27 = np.where(range_27 > 0, (df['C'] - llv_27) / range_27 * 100, 0)
        sma1 = sma_cn(pd.Series(k_stoch_27, index=df.index), 5, 1)
        sma2 = sma_cn(sma1, 3, 1)
        inner_calc = 3 * sma1 - 2 * sma2
        df['涨跌'] = inner_calc.rolling(5).mean()
        
        # 3. 评分系统
        # 使用全局参数
        buy_weights = BUY_WEIGHTS
        buy_threshold = BUY_THRESHOLD
        
        # 主力吸货分数
        score1 = np.zeros(len(df))
        base_cond = df['主力吸货'] > 0
        cont_cond = (df['主力吸货'] > df['主力吸货'].shift(1)) & (df['主力吸货'].shift(1) > df['主力吸货'].shift(2))
        strength_cond = df['主力吸货'] > df['主力吸货'].rolling(20).mean()
        
        score1[base_cond] += 50
        score1[cont_cond] += 30
        score1[strength_cond] += 20
        df['主力吸货分数'] = score1
        
        # 风险分数
        score2 = np.zeros(len(df))
        score2[df['风险'] < 20] = 100
        score2[(df['风险'] >= 20) & (df['风险'] < 50)] = 70
        score2[(df['风险'] >= 50) & (df['风险'] < 80)] = 30
        df['风险分数'] = score2
        
        # 涨跌分数
        score3 = np.zeros(len(df))
        score3[(df['涨跌'] > 0) & (df['涨跌'] > df['涨跌'].shift(1))] = 100
        score3[(df['涨跌'] > 0) & (df['涨跌'] <= df['涨跌'].shift(1))] = 50
        df['涨跌分数'] = score3
        
        df['买入总分'] = (df['主力吸货分数'] * buy_weights['s1_main'] + 
                         df['风险分数'] * buy_weights['s2_risk'] + 
                         df['涨跌分数'] * buy_weights['s3_momentum'])
        
        # 4. 判断最新信号 (只看最后一行)
        last_idx = df.index[-1]
        last_score = df.loc[last_idx, '买入总分']
        
        if last_score >= buy_threshold:
            info = f"总分:{last_score:.1f} (主力:{df.loc[last_idx, '主力吸货分数']:.0f}, 风险:{df.loc[last_idx, '风险分数']:.0f}, 涨跌:{df.loc[last_idx, '涨跌分数']:.0f})"
            return True, last_score, info
            
        return False, last_score, ""

    except Exception as e:
        # print(f"Error calculating indicator: {e}")
        return False, 0, ""

# --- 数据获取与筛选 ---

def get_financial_data(pro, ts_code_list, period):
    """批量获取财务数据"""
    df_list = []
    chunk_size = API_CHUNK_SIZE
    print(f"Fetching financial data for {len(ts_code_list)} stocks (Period: {period})...")
    
    for i in range(0, len(ts_code_list), chunk_size):
        chunk = ts_code_list[i:i+chunk_size]
        codes = ",".join(chunk)
        try:
            # dt_netprofit_yoy: 扣非净利润同比增长率
            df = pro.fina_indicator(ts_code=codes, period=period, fields='ts_code,dt_netprofit_yoy')
            df_list.append(df)
            time.sleep(0.1) # 避免频控
        except Exception as e:
            print(f"Error fetching financial data for chunk {i}: {e}")
    
    if df_list:
        return pd.concat(df_list)
    return pd.DataFrame()

def get_a_stocks():
    """获取A股列表并进行基本面初筛 (Tushare版)"""
    print("Fetching A-share fundamental data via Tushare...")
    
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # 1. 获取最新交易日
        today = datetime.datetime.now()
        today_str = today.strftime('%Y%m%d')
        start_date = (today - datetime.timedelta(days=30)).strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='', start_date=start_date, end_date=today_str, is_open='1')
        
        if cal.empty:
            print("No trade calendar found.")
            return pd.DataFrame()
            
        last_trade_date = cal.iloc[-1]['cal_date']
        print(f"Latest trade date: {last_trade_date}")

        # 2. 获取每日指标 (PE, 市值)
        df_daily = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,pe_ttm,total_mv')
        
        # 3. 获取股票基础信息 (名称)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        
        # 4. 合并
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # 5. 初步过滤 (PE & 市值)
        subset = df[
            (df['total_mv'] * 10000 >= MIN_MARKET_CAP) & 
            (df['pe_ttm'] >= MIN_PE) & 
            (df['pe_ttm'] <= MAX_PE)
        ].copy()
        
        print(f"Stocks passed PE & Cap filter: {len(subset)}")
        
        if subset.empty:
            return subset

        # 6. 增加财务筛选 (扣非净利润正增长)
        # 计算最近的报告期
        year = today.year
        if today.month <= 4:
            period = f"{year-1}1231" # 去年年报
        elif today.month <= 8:
            period = f"{year}0331" # 一季报
        elif today.month <= 10:
            period = f"{year}0630" # 中报
        else:
            period = f"{year}0930" # 三季报
            
        df_fina = get_financial_data(pro, subset['ts_code'].tolist(), period)
        
        if not df_fina.empty:
            # 合并财务数据
            subset = pd.merge(subset, df_fina, on='ts_code', how='inner')
            # 筛选 dt_netprofit_yoy > 0
            subset = subset[subset['dt_netprofit_yoy'] > 0].copy()
            print(f"Stocks passed Financial filter (Positive Growth): {len(subset)}")
        else:
            print("Warning: No financial data fetched. Skipping financial filter.")

        return subset

    except Exception as e:
        print(f"Failed to fetch A-share list from Tushare: {e}")
        return pd.DataFrame()

def check_financials(symbol):
    """
    检查单个股票的负债率
    Akshare获取个股财务指标较慢，这里作为演示，若为了速度可跳过或仅对触发信号的股票复查
    """
    try:
        # 这里为了演示速度，暂时返回True。
        # 实际操作建议：获取到信号后再去查负债率，减少请求次数。
        return True
    except:
        return False

def run_scanner():
    daily_picks = []
    weekly_picks = []
    
    # 1. 获取 A 股符合基本面的列表
    a_stocks = get_a_stocks()
    
    if a_stocks.empty:
        print("No stocks found in fundamental scan.")
        return [], []

    # 2. 遍历列表
    process_list = a_stocks
    print(f"Starting technical scan for {len(process_list)} stocks...")
    
    ts.set_token(TUSHARE_TOKEN)
    
    # 日期范围: 过去 1.5 年 (确保周线有足够数据, 90周 approx 630 days)
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=SCAN_LOOKBACK_DAYS)).strftime('%Y%m%d')

    for i, (index, row) in enumerate(process_list.iterrows()):
        symbol = row['ts_code']
        name = row['name']
        
        if i % 50 == 0:
            print(f"Progress: {i}/{len(process_list)}...")

        try:
            # --- 日线分析 ---
            df_daily = ts.pro_bar(ts_code=symbol, adj='qfq', start_date=start_date, end_date=end_date, freq='D')
            is_buy_d, score_d, info_d = calculate_technical_signals(df_daily)
            
            # --- 周线分析 ---
            df_weekly = ts.pro_bar(ts_code=symbol, adj='qfq', start_date=start_date, end_date=end_date, freq='W')
            is_buy_w, score_w, info_w = calculate_technical_signals(df_weekly)
            
            # 构建基础信息字符串
            base_info = f"- **PE(TTM)**: {row['pe_ttm']}\n- **扣非净利增速**: {row['dt_netprofit_yoy']}%\n"
            
            if is_buy_d:
                msg = f"### 🚀 {name} ({symbol})\n{base_info}- 📅 **日线信号**: {info_d}\n---\n"
                print(f"Found Daily: {name} ({symbol})")
                daily_picks.append(msg)
                
            if is_buy_w:
                msg = f"### 🚀 {name} ({symbol})\n{base_info}- 📅 **周线信号**: {info_w}\n---\n"
                print(f"Found Weekly: {name} ({symbol})")
                weekly_picks.append(msg)
            
            time.sleep(0.02) # 避免频控
            
        except Exception as e:
            # print(f"Error processing {symbol}: {e}")
            continue

    return daily_picks, weekly_picks

# --- 推送模块 ---

def send_pushplus(content):
    if not PUSHPLUS_TOKEN:
        print("PushPlus Token is missing. Skipping push notification.")
        return
    
    print(f"Sending PushPlus notification... (Token: {PUSHPLUS_TOKEN[:4]}***)")
    url = 'http://www.pushplus.plus/send'
    data = {
        "token": PUSHPLUS_TOKEN,
        "title": f"选股日报 {datetime.date.today()}",
        "content": content,
        "template": "markdown"
    }
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"PushPlus Response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to send PushPlus notification: {e}")

def send_email(content):
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_RECEIVER):
        return
    
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header("量化选股器", 'utf-8')
    message['To'] = Header("投资者", 'utf-8')
    message['Subject'] = Header(f"选股日报 {datetime.date.today()}", 'utf-8')

    try:
        smtpObj = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        smtpObj.login(EMAIL_USER, EMAIL_PASS)
        smtpObj.sendmail(EMAIL_USER, [EMAIL_RECEIVER], message.as_string())
        print("Email sent successfully")
    except smtplib.SMTPException as e:
        print(f"Error sending email: {e}")

# --- 主入口 ---

if __name__ == "__main__":
    print("--- Starting Stock Scan ---")
    daily_picks, weekly_picks = run_scanner()
    
    if daily_picks or weekly_picks:
        # Markdown 头部
        msg_content = f"# 📈 选股日报 {datetime.date.today()}\n\n"
        
        if daily_picks:
            msg_content += "## ☀️ 日线级别机会\n"
            msg_content += "".join(daily_picks)
            msg_content += "\n"
            
        if weekly_picks:
            msg_content += "## 📅 周线级别机会\n"
            msg_content += "".join(weekly_picks)
            msg_content += "\n"
        
        print("Sending notifications...")
        send_pushplus(msg_content)
        
        # 邮件发送
        send_email(msg_content)
    else:
        print("No stocks matched criteria today.")
        # 可选：也发送一个“今日无信号”的通知
        send_pushplus("今日无符合条件的股票信号。")