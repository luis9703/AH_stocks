import requests
import os

def test_pushplus():
    print("--- 开始测试 PushPlus 推送 ---")
    
    # 1. 获取 Token
    # 尝试从环境变量获取，如果没有，请用户手动输入或在代码中填入
    token = os.environ.get("PUSHPLUS_TOKEN")
    
    # 如果环境变量没设置，这里硬编码一个占位符，请用户替换
    if not token:
        # ！！！请将下方的 "你的Token" 替换为您真实的 PushPlus Token ！！！
        token = "7b4ee9c6a01c42009c59e7b1e193b108" 
    
    print(f"使用的 Token: {token}")
    
    if token == "你的Token" or not token:
        print("错误：Token 无效。请在代码中填入正确的 PushPlus Token，或设置环境变量 PUSHPLUS_TOKEN。")
        return

    # 2. 构造请求
    url = 'http://www.pushplus.plus/send'
    content = "这是一条来自 Stock Scanner 的测试消息。<br/>如果收到此消息，说明推送配置正确。"
    
    data = {
        "token": token,
        "title": "PushPlus 测试消息",
        "content": content,
        "template": "html"
    }
    
    print(f"正在发送请求到: {url}")
    
    try:
        # 3. 发送请求
        response = requests.post(url, json=data, timeout=10)
        
        # 4. 打印结果
        print(f"HTTP 状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            json_res = response.json()
            if json_res.get("code") == 200:
                print("✅ 推送请求成功！请检查微信是否收到消息。")
            else:
                print(f"❌ 推送请求被拒绝。错误码: {json_res.get('code')}, 错误信息: {json_res.get('msg')}")
        else:
            print("❌ HTTP 请求失败。")
            
    except Exception as e:
        print(f"❌ 发送过程中发生异常: {e}")

if __name__ == "__main__":
    test_pushplus()
