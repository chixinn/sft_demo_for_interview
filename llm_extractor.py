def test_api():
    """测试API连接"""
    
    print("\n" + "="*60)
    print("🧪 FRIDAY API 连接测试")
    print("="*60)
    
    # 1. 检查环境变量
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
    model_name = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")
    
    print(f"\n📋 配置信息:")
    print(f"  • API URL: {api_url}")
    print(f"  • 模型: {model_name}")
    print(f"  • API Key: {api_key[:20] if api_key else '未设置'}...")
    
    if not api_key or api_key == "your-api-key":
        print("\n❌ 错误: 未设置有效的 API Key")
        print("\n设置方法:")
        print("  export LLM_API_KEY='你的真实AppID'")
        print("\n获取 AppID:")
        print("  1. 访问: https://friday.sankuai.com/budget")
        print("  2. 创建租户 -> 申请大语言模型服务")
        print("  3. 获取 AppID")
        return False
    
    # 2. 测试API连接
    print(f"\n🔄 正在测试API连接...")
    
    try:
        import httpx
        import uuid
        import json
        
        trace_id = str(uuid.uuid4())
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "M-TraceId": trace_id
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "你好，请说一句话证明你正常工作"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "user": trace_id
                }
            )
            
            # 检查状态码
            print(f"\n📊 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ API连接成功！")
                
                result = response.json()
                
                # 显示响应内容
                if "choices" in result:
                    content = result["choices"][0]["message"]["content"]
                    print(f"\n💬 模型回复:")
                    print(f"  {content}")
                
                # 显示Token使用情况
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\n📈 Token使用:")
                    print(f"  • 输入: {usage.get('prompt_tokens', 0)}")
                    print(f"  • 输出: {usage.get('completion_tokens', 0)}")
                    print(f"  • 总计: {usage.get('total_tokens', 0)}")
                
                # 显示响应头（包含TraceId）
                print(f"\n🔍 调试信息:")
                print(f"  • TraceId: {trace_id}")
                print(f"  • 响应ID: {result.get('id', 'N/A')}")
                print(f"  • 模型: {result.get('model', 'N/A')}")
                
                print("\n" + "="*60)
                print("🎉 测试完成！API工作正常")
                print("="*60)
                return True
                
            elif response.status_code == 401:
                print("❌ 认证失败: API Key 无效")
                print("   请检查你的 AppID 是否正确")
                return False
                
            elif response.status_code == 450:
                print("❌ 请求内容安全审核失败")
                print("   请检查输入内容是否合规")
                return False
                
            elif response.status_code == 451:
                print("❌ 模型响应内容审核失败")
                print("   模型生成的回复未通过安全检查")
                return False
                
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(f"   响应内容: {response.text[:200]}")
                return False
                
    except ImportError:
        print("❌ 未安装 httpx")
        print("   安装: python3 -m pip install httpx")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


# ==========================================
# 2. 大模型特征提取模块 (LLM Extractor)
# 处理非结构化文本（如人行征信报文、尽调报告）
# ==========================================
class LLMFeatureExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        # 实际业务中，这里会初始化对应的 SDK，如 openai.OpenAI(api_key=...)
        
    def extract_hidden_risks(self, raw_text: str) -> dict:
        """
        调用外部大模型 API 解析长文本，提取风险特征
        """
        print("⏳ [Step 1] 正在调用云端大模型解析数十页的人行征信报文...")
        
        prompt = f"""
        作为资深风控专家，请阅读以下人行征信报文片段，提取该客户的隐性风险特征。
        请严格以 JSON 格式输出，包含以下三个字段：
        1. "capital_thirst" (资金饥渴度：低/中/高/极高)
        2. "guarantee_risk" (担保风险度：低/中/高/极高)
        3. "risk_summary" (一句话风险总结)
        
        【征信报文内容】：
        {raw_text}
        """
        
        # ⚠️ 模拟大模型 API 返回结果 (为了代码可直接运行，这里直接 mock 返回)
        # 实际代码应为：response = client.chat.completions.create(...) 
        # 然后解析 response.choices[0].message.content
        mock_llm_response = {
            "capital_thirst": "极高",
            "guarantee_risk": "高",
            "risk_summary": "存在严重多头借贷倾向，且卷入不良连带担保网络。"
        }
        
        print(f"✅ [Step 1 完成] 大模型解析结果: {mock_llm_response}")
        return mock_llm_response
