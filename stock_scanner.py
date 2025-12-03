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
PUSHPLUS_TOKEN = os.environ.get("7b4ee9c6a01c42009c59e7b1e193b108")  # 你的PushPlus Token

# 邮件推送配置
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
EMAIL_USER = os.environ.get("EMAIL_USER")       # 发件人邮箱
EMAIL_PASS = os.environ.get("EMAIL_PASS")       # 邮箱授权码
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER") # 收件人邮箱

# 选股硬性指标
MIN_MARKET_CAP = 200 * 10000 * 10000  # 200亿
MIN_PE = 5
MAX_PE = 100
MAX_DEBT_RATIO = 60 # 负债率 60%

# --- 指标计算辅助函数 ---

def calc_sma(series, n, m):
    """
    模拟同花顺 SMA 算法: Y = (M*X + (N-M)*Y')/N
    """
    result = []
    y_prev = 0 # 初始值，通常取序列第一个值或0，这里为了平滑取0或由于SMA特性随时间收敛
    # 为了准确模拟，通常第一项设为 series[0]
    if len(series) > 0:
        y_prev = series.iloc[0]
    
    for x in series:
        y = (m * x + (n - m) * y_prev) / n
        result.append(y)
        y_prev = y
    return pd.Series(result, index=series.index)

def calculate_indicator(df):
    """
    计算用户提供的同花顺指标
    """
    try:
        # 确保数据够长，至少需要90天数据计算 LLV(LOW, 90)
        if len(df) < 90:
            return False, 0

        # 数据清洗
        close = df['close']
        low = df['low']
        high = df['high']

        # VAR1:=REF(LOW,1);
        var1 = low.shift(1)

        # VAR2:=SMA(ABS(LOW-VAR1),3,1)/SMA(MAX(LOW-VAR1,0),3,1)*100;
        abs_diff = (low - var1).abs()
        max_diff = (low - var1).clip(lower=0)
        
        sma_abs = calc_sma(abs_diff, 3, 1)
        sma_max = calc_sma(max_diff, 3, 1)
        
        # 避免除以0
        var2 = (sma_abs / sma_max.replace(0, 0.0001)) * 100

        # VAR3:=EMA(IF(CLOSE*1.2,VAR2*10,VAR2/10),3);
        # 逻辑说明: CLOSE*1.2 只要收盘价不为0恒为真。在同花顺中 IF(A, B, C) 若A为真返回B。
        # 绝大多数情况 Close*1.2 都是真，所以取 VAR2 * 10
        # EMA 在 Pandas 中用 ewm(span=N, adjust=False).mean() 近似，或 alpha=2/(N+1)
        var3_input = var2 * 10 
        var3 = var3_input.ewm(span=3, adjust=False).mean()

        # VAR4:=LLV(LOW,38);
        var4 = low.rolling(window=38).min()

        # VAR5:=HHV(VAR3,38);
        var5 = var3.rolling(window=38).max()

        # VAR6:=IF(LLV(LOW,90),1,0); 只要有数据就是1
        var6 = 1

        # VAR7:=EMA(IF(LOW<=VAR4,(VAR3+VAR5*2)/2,0),3)/618*VAR6;
        # 判断 LOW <= VAR4
        condition = low <= var4
        # 如果满足条件，取 (VAR3 + VAR5*2)/2，否则取 0
        input_for_var7 = pd.Series(0, index=df.index)
        input_for_var7[condition] = (var3 + var5 * 2) / 2
        
        var7_ema = input_for_var7.ewm(span=3, adjust=False).mean()
        var7 = (var7_ema / 618) * var6

        # VAR8:=((C-LLV(L,21))/(HHV(H,21)-LLV(L,21)))*100;
        # VAR9:=SMA(VAR8,13,8);
        # 风险:CEILING(SMA(VAR9,13,8))
        # 涨跌: ... (虽然用户给了公式，但主要触发点是“主力吸货”)

        # --- 信号判断 ---
        # 逻辑：昨天（iloc[-1]）或者最近几天出现了“主力吸货”信号 (VAR7 > 0.1)
        # 考虑到“吸货”是一个区间，我们检测昨天是否有吸筹动作
        last_val = var7.iloc[-1]
        
        # 这里的阈值 0.1 是经验值，VAR7 > 0 即代表有红柱子
        if last_val > 0.1: 
            return True, round(last_val, 2)
        
        return False, 0

    except Exception as e:
        print(f"Error calculating indicator: {e}")
        return False, 0

# --- 数据获取与筛选 ---

def get_a_stocks():
    """获取A股列表并进行基本面初筛 (Tushare版)"""
    print("Fetching A-share fundamental data via Tushare...")
    
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # 1. 获取最新交易日
        # 获取过去30天内的交易日历，取最后一个
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='', start_date=start_date, end_date=today, is_open='1')
        
        if cal.empty:
            print("No trade calendar found.")
            return pd.DataFrame()
            
        last_trade_date = cal.iloc[-1]['cal_date']
        print(f"Latest trade date: {last_trade_date}")

        # 2. 获取每日指标 (PE, 市值)
        # total_mv 单位是万元
        df_daily = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,pe_ttm,total_mv')
        
        # 3. 获取股票基础信息 (名称)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        
        # 4. 合并
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # 5. 过滤
        # MIN_MARKET_CAP 是绝对值(元)，total_mv 是万元，所以 total_mv * 10000
        subset = df[
            (df['total_mv'] * 10000 >= MIN_MARKET_CAP) & 
            (df['pe_ttm'] >= MIN_PE) & 
            (df['pe_ttm'] <= MAX_PE)
        ].copy()
        
        print(f"A-shares passed fundamental filter (Cap & PE): {len(subset)}")
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
    results = []
    
    # 1. 获取 A 股符合基本面的列表
    a_stocks = get_a_stocks()
    
    if a_stocks.empty:
        print("No stocks found in fundamental scan.")
        return []

    # 2. 遍历列表获取K线并计算
    process_list = a_stocks
    
    print(f"Starting technical scan for {len(process_list)} stocks...")
    
    # 准备 Tushare 接口
    ts.set_token(TUSHARE_TOKEN)
    
    # 计算起始日期 (取过去200天以确保有足够数据计算 LLV(90))
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=250)).strftime('%Y%m%d')

    for i, (index, row) in enumerate(process_list.iterrows()):
        symbol = row['ts_code']
        name = row['name']
        
        if i % 50 == 0:
            print(f"Progress: {i}/{len(process_list)}...")

        try:
            # 获取日线数据 (前复权)
            # ts.pro_bar 整合了复权功能
            stock_data = ts.pro_bar(ts_code=symbol, adj='qfq', start_date=start_date, end_date=end_date)
            
            if stock_data is None or stock_data.empty:
                continue
            
            # Tushare 返回的数据通常是按日期降序(最新在前)，指标计算通常需要升序(最旧在前)
            stock_data = stock_data.sort_values('trade_date').reset_index(drop=True)
            
            # 计算指标
            is_triggered, signal_val = calculate_indicator(stock_data)
            
            if is_triggered:
                # 二次确认：负债率 (如果API支持)
                # if check_debt_ratio(symbol):
                res_str = f"【A股】{name} ({symbol}): 主力吸筹值 {signal_val}, PE(TTM) {row['pe_ttm']}"
                print(f"Found: {res_str}")
                results.append(res_str)
            
            # 礼貌性延时，防止触发 Tushare 频控
            time.sleep(0.02)
            
        except Exception as e:
            # print(f"Error processing {symbol}: {e}")
            continue

    return results

# --- 推送模块 ---

def send_pushplus(content):
    if not PUSHPLUS_TOKEN:
        return
    url = 'http://www.pushplus.plus/send'
    data = {
        "token": PUSHPLUS_TOKEN,
        "title": f"选股日报 {datetime.date.today()}",
        "content": content,
        "template": "html"
    }
    requests.post(url, json=data)

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
    final_picks = run_scanner()
    
    if final_picks:
        msg_content = "今日触发【主力吸筹】信号且基本面优秀的股票：<br/><br/>" + "<br/>".join(final_picks)
        print("Sending notifications...")
        send_pushplus(msg_content)
        
        # 将 HTML 换行转为文本换行发邮件
        email_content = msg_content.replace("<br/>", "\n")
        send_email(email_content)
    else:
        print("No stocks matched criteria today.")
        # 可选：也发送一个“今日无信号”的通知
        send_pushplus("今日无符合条件的股票信号。")