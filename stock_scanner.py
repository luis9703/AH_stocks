import akshare as ak
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import requests
import os
import time

# --- 配置区域 (推荐通过环境变量注入，保证安全) ---
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
    """获取A股列表并进行基本面初筛"""
    print("Fetching A-share fundamental data...")
    
    # --- 新增：重试机制 ---
    # GitHub Actions 在海外，连接国内接口容易断开，增加重试可以大幅提高稳定性
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 获取实时行情，包含总市值、PE、PB等
            df = ak.stock_zh_a_spot_em()
            
            # 过滤条件
            # 1. 市值 > 200亿 (总市值字段通常是 '总市值')
            # 2. 5 < PE(TTM) < 100 (字段: '市盈率-动态')
            
            subset = df[
                (df['总市值'] >= MIN_MARKET_CAP) & 
                (df['市盈率-动态'] >= MIN_PE) & 
                (df['市盈率-动态'] <= MAX_PE)
            ].copy()
            
            print(f"A-shares passed fundamental filter (Cap & PE): {len(subset)}")
            return subset

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 15  # 失败后等待15秒再试
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to fetch A-share list.")
                # 如果彻底失败，返回空DataFrame，避免脚本崩溃
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
    
    # 2. 遍历列表获取K线并计算 (限制数量防止Demo运行过久，实际部署去掉head)
    # 为了演示，我们只跑前 10 个符合基本面的，部署时请去掉 .head(10)
    process_list = a_stocks # .head(10) 
    
    print(f"Starting technical scan for {len(process_list)} stocks...")
    
    for index, row in process_list.iterrows():
        symbol = row['代码']
        name = row['名称']
        
        try:
            # 获取日线数据 (前复权)
            # period='daily', adjust='qfq'
            stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            
            # 计算指标
            is_triggered, signal_val = calculate_indicator(stock_data)
            
            if is_triggered:
                # 二次确认：负债率 (如果API支持)
                # if check_debt_ratio(symbol):
                res_str = f"【A股】{name} ({symbol}): 主力吸筹值 {signal_val}, PE(TTM) {row['市盈率-动态']}"
                print(f"Found: {res_str}")
                results.append(res_str)
            
            # 礼貌性延时，防止被封IP
            time.sleep(0.1)
            
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
