# -*- coding: utf-8 -*-
"""
大模型知识蒸馏批处理脚本：逆向生成风控 CoT 思维链
输入：历史宽表 (DataFrame) + 非结构化数据 + 真实风险Y标 + 真实批核Y标
输出：可直接用于小模型微调的 ShareGPT 格式 dataset.jsonl

使用美团 FRIDAY API (https://friday.sankuai.com)
"""

import pandas as pd
import json
import time
import os
import asyncio
import httpx
import uuid
from tqdm import tqdm

# ==========================================
# 1. 基础配置 (美团 FRIDAY API)
# ==========================================
# 获取 AppID: https://friday.sankuai.com/budget
API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
API_URL = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")

OUTPUT_FILE = "sft_dataset.jsonl"
MAX_RETRIES = 3
REQUEST_DELAY = 0.5  # 请求间隔(秒)，防止QPS过高

# ==========================================
# 2. 核心：构造 Prompt
# ==========================================
def build_prompt_from_row(row):
    """
    根据每一行数据，构造给大模型的 Prompt。
    核心逻辑：结合【真实批核Y标】和【真实风险Y标】，告诉大模型“正确的决定”是什么。
    """
    # 提取特征
    structured_data = f"月收入={row['income']}, 负债比={row['debt_ratio']}, 历史逾期={row['history_overdue']}"
    unstructured_data = row['unstructured_report']
    
    # 【风控业务逻辑对齐】：
    # 假设 real_risk_Y 中，0=好客户(未逾期)，1=坏客户(逾期)
    # 哪怕历史 real_approval_Y 是通过的，只要他后来逾期了(1)，我们在微调时也要教小模型去【拒绝】他！
    ideal_decision = "REJECT" if row['real_risk_Y'] == 1 else "APPROVE"
    
    # 构造系统提示词（设定大模型的人设和输出规范）
    system_prompt = (
        "你是一个由顶尖商业银行聘请的资深信贷风控专家。"
        "你的任务是根据客户的结构化特征和非结构化征信记录，对已知的【目标决策】进行严密的逆向逻辑推导。"
        "请务必输出严格的 JSON 格式，包含三个字段：\n"
        "1. 'reasoning_process': 一段不少于50字的思维链推导过程，说明为什么做出该决策。\n"
        "2. 'credit_score': 预估的信用分(0-1000，拒绝通常低于500)。\n"
        "3. 'decision': 最终决定(只能是 APPROVE 或 REJECT)。"
    )
    
    # 构造用户输入（给到小模型微调时的真正 Input）
    user_input = (
        f"请评估该客户：\n"
        f"【结构化特征】：{structured_data}\n"
        f"【非结构化特征】：{unstructured_data}"
    )
    
    # 构造蒸馏指令（仅给大模型看，告诉它目标是什么）
    distill_instruction = (
        f"\n\n[内部指令]：结合上述特征，该客户的真实表现已被验证。请基于目标决策【{ideal_decision}】，"
        f"逆向推导出一套符合风控逻辑的判断理由，并以JSON格式输出。"
    )

    return system_prompt, user_input, user_input + distill_instruction

# ==========================================
# 3. API 调用与重试机制 (美团 FRIDAY API)
# ==========================================
async def _call_friday_api_async(system_prompt: str, user_prompt: str) -> str:
    """异步调用美团 FRIDAY API"""
    trace_id = str(uuid.uuid4())
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "M-TraceId": trace_id
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
                "user": trace_id
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API错误: {response.status_code} - {response.text[:200]}")


def fetch_cot_from_llm(system_prompt, distill_prompt):
    """调用美团 FRIDAY API 并解析返回的 JSON"""
    # 检查 API Key
    if API_KEY == "your-api-key":
        print("\n❌ 未配置 API Key，请设置环境变量: export LLM_API_KEY='你的AppID'")
        return None, None
    
    for attempt in range(MAX_RETRIES):
        try:
            # 调用 API
            result_str = asyncio.run(_call_friday_api_async(system_prompt, distill_prompt))
            
            # 提取 JSON（处理模型可能输出 markdown 代码块的情况）
            json_str = result_str
            if "```json" in result_str:
                start = result_str.find("```json") + 7
                end = result_str.find("```", start)
                json_str = result_str[start:end].strip()
            elif "```" in result_str:
                start = result_str.find("```") + 3
                end = result_str.find("```", start)
                json_str = result_str[start:end].strip()
            elif "{" in result_str and "}" in result_str:
                start = result_str.find("{")
                end = result_str.rfind("}") + 1
                json_str = result_str[start:end]
            
            # 校验是否为合法 JSON
            parsed_json = json.loads(json_str)
            return json_str, parsed_json
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️ 第 {attempt + 1} 次JSON解析失败: {e}")
            print(f"   原始响应: {result_str[:200]}...")
            time.sleep(1)
            
        except Exception as e:
            print(f"\n⚠️ 第 {attempt + 1} 次调用失败: {e}")
            time.sleep(2)
            
    return None, None  # 重试全部失败则返回 None

# ==========================================
# 4. 批处理主干与持久化存储
# ==========================================
def process_dataframe(df):
    """遍历 DataFrame，蒸馏数据并实时追加写入 JSONL"""
    print(f"🚀 开始跑批，共计 {len(df)} 条进件数据...")
    
    success_count = 0
    # 打开文件，使用 'a' (append) 模式，防止程序崩溃导致数据全部丢失
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            
            system_prompt, user_input, distill_prompt = build_prompt_from_row(row)
            
            # 调用大模型生成 CoT
            cot_json_str, parsed_json = fetch_cot_from_llm(system_prompt, distill_prompt)
            
            if cot_json_str:
                # 组装成微调框架(如 LLaMA-Factory)所需的 ShareGPT/ChatML 格式
                sft_record = {
                    "messages": [
                        {"role": "system", "content": "你是一个专业的信贷风控决策引擎。你的任务是根据用户的输入特征，进行严谨的逻辑推理，并严格按照 JSON 格式输出决策结果。决不能编造数据。"},
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": cot_json_str}
                    ]
                }
                # 写入文件，每生成一条保存一条（断点续传的精髓）
                f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
                success_count += 1
                
            # 基础限流：防止 QPS 过高被封 API
            time.sleep(REQUEST_DELAY) 
            
    print(f"🎉 跑批完成！成功蒸馏 {success_count}/{len(df)} 条 SFT 数据。")
    print(f"📁 数据已保存至: {OUTPUT_FILE}")

# ==========================================
# 5. 执行入口
# ==========================================
def check_config():
    """检查配置状态"""
    print("\n" + "="*60)
    print("📋 美团 FRIDAY API 配置检查")
    print("="*60)
    print(f"\n• API URL: {API_URL}")
    print(f"• 模型: {MODEL_NAME}")
    print(f"• API Key: {API_KEY[:20] if API_KEY != 'your-api-key' else '未设置'}...")
    
    if API_KEY == "your-api-key":
        print("\n❌ 未配置 API Key!")
        print("\n📝 配置方法:")
        print("   export LLM_API_KEY='你的AppID'")
        print("\n🔗 获取 AppID: https://friday.sankuai.com/budget")
        return False
    return True


def load_data_from_csv(csv_path: str = None) -> pd.DataFrame:
    """从 CSV 文件加载数据，或使用模拟数据"""
    if csv_path and os.path.exists(csv_path):
        print(f"📥 从 {csv_path} 加载数据...")
        return pd.read_csv(csv_path)
    
    # 使用模拟数据
    print("📥 使用模拟数据进行测试...")
    mock_data = {
        "apply_id": ["APP-001", "APP-002", "APP-003", "APP-004", "APP-005"],
        "income": [28000, 8500, 32000, 15000, 45000],
        "debt_ratio": [0.25, 0.65, 0.30, 0.45, 0.15],
        "history_overdue": [0, 2, 0, 1, 0],
        "unstructured_report": [
            "人行征信显示：无任何不良记录，近三个月无网贷查询。工作稳定，已婚有房。",
            "人行征信异常：近1个月内存在12次小贷公司审批查询，且有一笔 5万元连带担保，该企业已被列为失信被执行人。",
            "补充调查：客户为某世界500强企业中层管理，收入稳定，名下有房产两套。",
            "征信记录：近6个月有3次逾期记录，最高逾期天数45天。信用卡使用率90%，存在以卡养卡嫌疑。",
            "人行征信良好：无逾期记录，近半年仅有1次正常贷款查询。职业为公立医院医生，工作10年。"
        ],
        "real_approval_Y": ["APPROVE", "APPROVE", "REJECT", "REJECT", "APPROVE"],
        "real_risk_Y": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(mock_data)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 大模型知识蒸馏 - CoT 思维链生成")
    print("="*60)
    
    # 检查配置
    if not check_config():
        print("\n⚠️ 请先配置 API Key 后再运行")
        exit(1)
    
    # 加载数据
    df_raw = load_data_from_csv()
    print(f"\n📊 数据概览:")
    print(f"   • 总样本数: {len(df_raw)}")
    print(f"   • 风险分布: {df_raw['real_risk_Y'].value_counts().to_dict()}")
    print(f"   • 审批分布: {df_raw['real_approval_Y'].value_counts().to_dict()}")
    
    # 清理旧文件
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"\n🗑️ 已清理旧文件: {OUTPUT_FILE}")
    
    # 执行批处理
    print(f"\n" + "="*60)
    process_dataframe(df_raw)