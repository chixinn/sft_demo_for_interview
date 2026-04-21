# -*- coding: utf-8 -*-
"""
轻量版测试脚本 - 仅测试美团 FRIDAY API 调用
不需要安装 torch/transformers 等大模型依赖
"""

import json
import os
import asyncio
import httpx
import uuid

# ==========================================
# 配置模块
# ==========================================
class Config:
    # 美团 FRIDAY API 配置
    # 获取 AppID: https://friday.sankuai.com/budget
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")


# ==========================================
# LLM 特征提取模块
# ==========================================
class LLMFeatureExtractor:
    def __init__(self, api_key=None, api_url=None, model=None):
        self.api_key = api_key or Config.LLM_API_KEY
        self.api_url = api_url or Config.LLM_API_URL
        self.model = model or Config.LLM_MODEL
        
        if self.api_key == "your-api-key":
            print("⚠️ 警告: 未设置 API Key，将使用模拟模式运行")
            print("   请设置环境变量: export LLM_API_KEY='你的AppID'")
            print("   获取 AppID: https://friday.sankuai.com/budget")
            self._mock_mode = True
        else:
            self._mock_mode = False
            print(f"✅ FRIDAY API 已配置")
            print(f"   URL: {self.api_url}")
            print(f"   模型: {self.model}")
        
    async def _call_api(self, prompt: str) -> str:
        """调用美团 FRIDAY API"""
        trace_id = str(uuid.uuid4())
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "M-TraceId": trace_id
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "user": trace_id
                }
            )
            
            print(f"📊 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # 显示 Token 使用情况
                if "usage" in result:
                    usage = result["usage"]
                    print(f"📈 Token使用: 输入={usage.get('prompt_tokens', 0)}, 输出={usage.get('completion_tokens', 0)}")
                
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text[:200]}")
    
    def extract_hidden_risks(self, raw_text: str) -> dict:
        """
        调用美团 FRIDAY API 解析长文本，提取风险特征
        """
        print("⏳ [Step 1] 正在调用 FRIDAY 大模型解析征信报文...")
        
        prompt = f"""作为资深风控专家，请阅读以下人行征信报文片段，提取该客户的隐性风险特征。
请严格以 JSON 格式输出，包含以下三个字段：
1. "capital_thirst" (资金饥渴度：低/中/高/极高)
2. "guarantee_risk" (担保风险度：低/中/高/极高)
3. "risk_summary" (一句话风险总结)

只需输出 JSON，不要有其他内容。

【征信报文内容】：
{raw_text}"""
        
        if self._mock_mode:
            # 模拟模式：返回预设结果
            mock_llm_response = {
                "capital_thirst": "极高",
                "guarantee_risk": "高",
                "risk_summary": "存在严重多头借贷倾向，且卷入不良连带担保网络。"
            }
            print(f"✅ [Step 1 完成] 大模型解析结果 (模拟): {mock_llm_response}")
            return mock_llm_response
        
        try:
            # 真实调用 API
            response_text = asyncio.run(self._call_api(prompt))
            print(f"📥 API 原始响应: {response_text}")
            
            # 解析 JSON 响应
            json_str = response_text
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            
            result = json.loads(json_str)
            print(f"✅ [Step 1 完成] 大模型解析结果: {result}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            return {
                "capital_thirst": "中",
                "guarantee_risk": "中",
                "risk_summary": response_text[:100] if 'response_text' in dir() else "解析失败"
            }
        except Exception as e:
            print(f"❌ API调用出错: {e}")
            return {
                "capital_thirst": "高",
                "guarantee_risk": "高",
                "risk_summary": f"API调用异常: {str(e)[:50]}"
            }


def check_api_config():
    """检查 API 配置状态"""
    print("\n" + "="*60)
    print("📋 美团 FRIDAY API 配置检查")
    print("="*60)
    
    api_key = Config.LLM_API_KEY
    print(f"\n• API URL: {Config.LLM_API_URL}")
    print(f"• 模型: {Config.LLM_MODEL}")
    print(f"• API Key: {api_key[:20] if api_key and api_key != 'your-api-key' else '未设置'}...")
    
    if api_key == "your-api-key":
        print("\n❌ 未配置 API Key!")
        print("\n📝 配置方法:")
        print("   export LLM_API_KEY='你的AppID'")
        print("\n🔗 获取 AppID:")
        print("   1. 访问: https://friday.sankuai.com/budget")
        print("   2. 创建租户 -> 申请大语言模型服务")
        print("   3. 复制 AppID")
        print("\n💡 当前将使用【模拟模式】运行，结果为预设值")
        return False
    return True


def main():
    print("\n" + "="*60)
    print("🚀 轻量版测试 - 美团 FRIDAY API 调用")
    print("="*60)
    
    # 检查 API 配置
    api_configured = check_api_config()
    
    # 模拟业务系统传入的客户进件数据
    unstructured_credit_report = """
    ... (前面省略5000字常规信息) ...
    特别记录：客户近 3 个月内有 18 次各类消费金融和小贷公司的贷款审批查询记录。
    担保信息：客户作为保证人，为某关联企业提供 50 万担保，该企业近期被法院列为被执行人。
    ... (后面省略3000字) ...
    """
    
    # 初始化组件
    extractor = LLMFeatureExtractor()
    
    print("\n>>> 开始处理进件编号：APP-2023-9981 <<<")
    
    # 调用 API 提取风险特征
    extracted_features = extractor.extract_hidden_risks(unstructured_credit_report)
    
    print("\n" + "="*60)
    print("🎉 【处理完毕】提取的风险特征：")
    print("-" * 40)
    print(json.dumps(extracted_features, ensure_ascii=False, indent=2))
    print("="*60)
    
    if not api_configured:
        print("\n💡 提示: 设置 LLM_API_KEY 后可使用真实 API 调用")
        print("   export LLM_API_KEY='你的AppID'")


if __name__ == "__main__":
    main()
