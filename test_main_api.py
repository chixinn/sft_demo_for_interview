# -*- coding: utf-8 -*-
"""
双模态智能信贷审批流水线 - 纯 API 版本
架构：FRIDAY API 大模型(特征提取) + FRIDAY API 大模型(综合决策)

无需本地模型依赖，完全使用美团 FRIDAY API
"""

import json
import os
import asyncio
import httpx
import uuid

# ==========================================
# 1. 配置模块 (Config)
# ==========================================
class Config:
    # 美团 FRIDAY API 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")


# ==========================================
# 2. FRIDAY API 客户端
# ==========================================
class FridayClient:
    """美团 FRIDAY API 统一客户端"""
    
    def __init__(self, api_key=None, api_url=None, model=None):
        self.api_key = api_key or Config.LLM_API_KEY
        self.api_url = api_url or Config.LLM_API_URL
        self.model = model or Config.LLM_MODEL
        self._mock_mode = (self.api_key == "your-api-key")
        
        if self._mock_mode:
            print("⚠️ 未配置 API Key，将使用模拟模式")
        else:
            print(f"✅ FRIDAY API 已配置: {self.model}")
    
    async def _call_api(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """调用 FRIDAY API"""
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
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "user": trace_id
                }
            )
            
            print(f"📊 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                print(f"📈 Token: 输入={usage.get('prompt_tokens', 0)}, 输出={usage.get('completion_tokens', 0)}")
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text[:200]}")
    
    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """同步调用接口"""
        if self._mock_mode:
            return None
        return asyncio.run(self._call_api(prompt, temperature, max_tokens))


# ==========================================
# 3. 大模型特征提取模块 (Step 1)
# ==========================================
class LLMFeatureExtractor:
    """从非结构化文本提取风险特征"""
    
    def __init__(self, client: FridayClient):
        self.client = client
    
    def extract_hidden_risks(self, raw_text: str) -> dict:
        print("⏳ [Step 1] 正在调用 FRIDAY 大模型解析征信报文...")
        
        prompt = f"""作为资深风控专家，请阅读以下人行征信报文片段，提取该客户的隐性风险特征。
请严格以 JSON 格式输出，包含以下三个字段：
1. "capital_thirst" (资金饥渴度：低/中/高/极高)
2. "guarantee_risk" (担保风险度：低/中/高/极高)
3. "risk_summary" (一句话风险总结)

只需输出 JSON，不要有其他内容。

【征信报文内容】：
{raw_text}"""
        
        if self.client._mock_mode:
            result = {
                "capital_thirst": "极高",
                "guarantee_risk": "高",
                "risk_summary": "存在严重多头借贷倾向，且卷入不良连带担保网络。"
            }
            print(f"✅ [Step 1 完成] 模拟结果: {result}")
            return result
        
        try:
            response_text = self.client.call(prompt)
            print(f"📥 原始响应: {response_text[:100]}...")
            
            # 提取 JSON
            json_str = response_text
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            
            result = json.loads(json_str)
            print(f"✅ [Step 1 完成] 风险特征: {result}")
            return result
            
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            return {"capital_thirst": "中", "guarantee_risk": "中", "risk_summary": str(e)[:50]}


# ==========================================
# 4. 决策打分模块 (Step 2)
# ==========================================
class DecisionScorer:
    """综合决策打分"""
    
    def __init__(self, client: FridayClient):
        self.client = client
    
    def generate_decision(self, structured_data: dict, llm_features: dict) -> str:
        print("⏳ [Step 2] 正在调用 FRIDAY 大模型进行综合决策...")
        
        prompt = f"""你是一个信贷审批决策引擎。请结合以下双模态特征，给出审批决策。

【结构化特征】：
- 月收入: {structured_data['monthly_income']} 元
- 负债比: {structured_data['debt_ratio']}
- 本行逾期次数: {structured_data['local_overdue']}

【征信解析特征】：
- 资金饥渴度: {llm_features['capital_thirst']}
- 担保风险: {llm_features['guarantee_risk']}
- 风险点: {llm_features['risk_summary']}

请输出：
1. 信用评分 (0-1000分)
2. 审批建议 (通过/拒绝)
3. 决策理由 (一句话)

请按以下格式输出：
评分=XXX
审批建议=XXX
决策理由=XXX"""
        
        if self.client._mock_mode:
            result = "评分=410\n审批建议=拒绝\n决策理由=多头借贷风险高，存在不良担保网络。"
            print(f"✅ [Step 2 完成] 模拟结果")
            return result
        
        try:
            response_text = self.client.call(prompt, max_tokens=300)
            print(f"✅ [Step 2 完成] 决策结果已生成")
            return response_text
        except Exception as e:
            print(f"❌ 决策失败: {e}")
            return f"评分=500\n审批建议=人工审核\n决策理由=API调用异常"


# ==========================================
# 5. 主流水线
# ==========================================
def main():
    print("\n" + "="*60)
    print("🚀 双模态智能信贷审批流水线 (纯API版)")
    print("="*60)
    
    # 检查配置
    api_key = Config.LLM_API_KEY
    print(f"\n📋 配置信息:")
    print(f"   • API URL: {Config.LLM_API_URL}")
    print(f"   • 模型: {Config.LLM_MODEL}")
    print(f"   • API Key: {api_key[:20] if api_key != 'your-api-key' else '未设置'}...")
    
    # 初始化客户端
    client = FridayClient()
    
    # 模拟进件数据
    structured_data = {
        "monthly_income": 25000,
        "debt_ratio": 0.3,
        "local_overdue": 0
    }
    
    unstructured_credit_report = """
    ... (前面省略5000字常规信息) ...
    特别记录：客户近 3 个月内有 18 次各类消费金融和小贷公司的贷款审批查询记录。
    担保信息：客户作为保证人，为某关联企业提供 50 万担保，该企业近期被法院列为被执行人。
    ... (后面省略3000字) ...
    """
    
    print("\n>>> 开始处理进件编号：APP-2023-9981 <<<")
    print("="*60)
    
    # Step 1: 特征提取
    extractor = LLMFeatureExtractor(client)
    extracted_features = extractor.extract_hidden_risks(unstructured_credit_report)
    
    print()
    
    # Step 2: 综合决策
    scorer = DecisionScorer(client)
    decision_result = scorer.generate_decision(structured_data, extracted_features)
    
    # 输出最终结果
    print("\n" + "="*60)
    print("🎉 【流水线执行完毕】最终信贷决策结果：")
    print("-" * 40)
    print(decision_result)
    print("="*60)
    
    # 输出结构化摘要
    print("\n📊 决策摘要:")
    print(f"   • 资金饥渴度: {extracted_features.get('capital_thirst', 'N/A')}")
    print(f"   • 担保风险: {extracted_features.get('guarantee_risk', 'N/A')}")
    print(f"   • 风险总结: {extracted_features.get('risk_summary', 'N/A')[:50]}...")


if __name__ == "__main__":
    main()
