# 配置文件
# -*- coding: utf-8 -*-
"""
双模态智能信贷审批流水线 (Dual-Modal Credit Approval Pipeline)
架构：千亿级API大模型(特征提取) + 边缘端小模型(Phi-1.5 SFT 综合决策)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. 配置模块 (Config)
# ==========================================
class Config:
    # 大模型 API 配置 (这里以通用大模型为例)
    LLM_API_KEY = "your_openai_or_baidu_api_key_here"
    
    # 小模型 (SFT) 路径配置
    BASE_MODEL_PATH = "microsoft/phi-1_5"  # 基础小模型
    LORA_WEIGHTS_PATH = "./lora_weights"   # 你之前 SFT 训练好的 LoRA 权重路径
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SEQ_LEN = 512


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
def main():
    print("="*60)
    print("🚀 开始执行：双模态智能信贷审批流水线")
    print("="*60)
    
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
    extractor = LLMFeatureExtractor(api_key=Config.LLM_API_KEY)
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

if __name__ == "__main__":
    main()