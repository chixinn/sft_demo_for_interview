# -*- coding: utf-8 -*-
"""
双模态智能信贷审批流水线 (Dual-Modal Credit Approval Pipeline)
架构：千亿级API大模型(特征提取) + 边缘端小模型(Phi-1.5 SFT 综合决策)

使用美团内部 FRIDAY API (https://friday.sankuai.com)
"""

import json
import os
import asyncio
import httpx
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. 配置模块 (Config)
# ==========================================
class Config:
    # ==========================================
    # 美团 FRIDAY API 配置
    # 获取 AppID: https://friday.sankuai.com/budget
    # ==========================================
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")  # 或 GLM-4-9B-Chat
    
    # 小模型 (SFT) 路径配置
    BASE_MODEL_PATH = "microsoft/phi-1_5"  # 基础小模型
    LORA_WEIGHTS_PATH = "./lora_weights"   # 你之前 SFT 训练好的 LoRA 权重路径
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SEQ_LEN = 512

# ==========================================
# 2. 大模型特征提取模块 (LLM Extractor)
# 处理非结构化文本（如人行征信报文、尽调报告）
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
                    "temperature": 0.1,  # 降低随机性，保证结果稳定
                    "max_tokens": 500,
                    "user": trace_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
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
            print(f"📥 API 原始响应: {response_text[:200]}...")
            
            # 解析 JSON 响应
            # 尝试提取 JSON 部分（处理模型可能输出额外文字的情况）
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
            print(f"   使用默认值")
            return {
                "capital_thirst": "中",
                "guarantee_risk": "中",
                "risk_summary": response_text[:100] if 'response_text' in dir() else "解析失败"
            }
        except Exception as e:
            print(f"❌ API调用出错: {e}")
            print("   使用模拟结果")
            return {
                "capital_thirst": "高",
                "guarantee_risk": "高",
                "risk_summary": f"API调用异常: {str(e)[:50]}"
            }

# ==========================================
# 3. 小模型决策打分模块 (SFT Scorer)
# 处理结构化数据 + 大模型提取的特征，给出最终决策
# ==========================================
class SmallModelScorer:
    def __init__(self):
        print(f"⏳ [初始化] 正在加载本地 SFT 小模型 ({Config.BASE_MODEL_PATH}) + LoRA权重...")
        try:
            # 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_PATH, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                Config.BASE_MODEL_PATH, 
                torch_dtype=torch.float32 if Config.DEVICE == "cpu" else torch.float16,
                device_map=Config.DEVICE,
                trust_remote_code=True
            )
            
            # 融合 LoRA 权重
            # 注意：如果你还没有训练好的权重，这里会 catch 异常并使用基础模型模拟
            self.model = PeftModel.from_pretrained(base_model, Config.LORA_WEIGHTS_PATH)
            print("✅ 本地 SFT 模型加载成功！")
            
        except Exception as e:
            print(f"⚠️ 提示: 未找到 LoRA 权重或模型 ({e})。系统将以【模拟模式】运行...")
            self.model = None

    def generate_decision(self, structured_data: dict, llm_features: dict) -> str:
        """
        融合双模态数据，通过 SFT 模型生成最终评分和拒批理由
        """
        print("⏳ [Step 2] 正在将多模态数据输入本地 SFT 小模型进行综合决策...")
        
        # 构建 SFT 训练时使用的 Prompt 模板
        instruction = (
            f"你是一个信贷审批决策引擎。请结合以下双模态特征，计算最终个人信用评分（0-1000分），并给出审批建议（通过/拒绝）及简短理由。\n"
            f"【结构化特征】：月收入={structured_data['monthly_income']}, 负债比={structured_data['debt_ratio']}, 本行逾期={structured_data['local_overdue']}\n"
            f"【征信解析特征】：资金饥渴度={llm_features['capital_thirst']}, 担保风险={llm_features['guarantee_risk']}, 风险点={llm_features['risk_summary']}"
        )
        
        prompt = f"<s>[INST] {instruction} [/INST] "
        
        if self.model is not None:
            # 真实模型推理逻辑
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    temperature=0.1, # 降低温度，保证决策的确定性
                    do_sample=False
                )
            result = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return result
        else:
            # 模拟 SFT 模型的理想输出结果 (便于你测试业务流)
            return "评分=410\n审批建议=拒绝\n决策理由=虽然客户表内收入和负债情况良好，但通过人行征信解析发现其近期频繁向外部小额贷款机构申请借款（多头借贷）；且其名下存在高风险的连带担保债务。综合判定违约概率极高，建议拦截。"

# ==========================================
# 4. 主干流水线 (Main Pipeline)
# ==========================================
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
        print("   3. 复制 AppID");
        print("\n💡 当前将使用【模拟模式】运行，结果为预设值")
        return False
    return True


def main():
    print("\n" + "="*60)
    print("🚀 开始执行：双模态智能信贷审批流水线")
    print("="*60)
    
    # 检查 API 配置
    api_configured = check_api_config()
    
    # 模拟业务系统传入的客户进件数据
    # 1. 业务系统查出的表格数据 (优质指标)
    structured_data = {
        "monthly_income": 25000,
        "debt_ratio": 0.3,
        "local_overdue": 0
    }
    
    # 2. 爬虫或OCR获取的长篇非结构化文本 (潜藏风险)
    unstructured_credit_report = """
    ... (前面省略5000字常规信息) ...
    特别记录：客户近 3 个月内有 18 次各类消费金融和小贷公司的贷款审批查询记录。
    担保信息：客户作为保证人，为某关联企业提供 50 万担保，该企业近期被法院列为被执行人。
    ... (后面省略3000字) ...
    """
    
    # 初始化组件
    extractor = LLMFeatureExtractor()
    scorer = SmallModelScorer()
    
    print("\n>>> 开始处理进件编号：APP-2023-9981 <<<")
    # 流水线执行
    # 第一步：大模型处理脏数据/长文本
    extracted_features = extractor.extract_hidden_risks(unstructured_credit_report)
    
    # 第二步：小模型结合全部特征进行毫秒级打分
    decision_result = scorer.generate_decision(structured_data, extracted_features)
    
    print("\n" + "="*60)
    print("🎉 【流水线执行完毕】最终信贷决策结果：")
    print("-" * 40)
    print(decision_result)
    print("="*60)
    
    if not api_configured:
        print("\n💡 提示: 设置 LLM_API_KEY 后可使用真实 API 调用")

if __name__ == "__main__":
    main()