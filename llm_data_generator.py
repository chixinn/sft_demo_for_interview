# -*- coding: utf-8 -*-
"""
使用 LLM 辅助构建 SFT 数据集
利用大模型的能力生成高质量训练样本
"""

import json
import os
import asyncio
import httpx
import uuid
from typing import List, Dict

# ==========================================
# 配置
# ==========================================
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
LLM_API_URL = os.getenv("LLM_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "LongCat-8B-128K-Chat")


class LLMDataGenerator:
    """使用 LLM 生成/增强训练数据"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or LLM_API_KEY
        self.api_url = LLM_API_URL
        self.model = LLM_MODEL
        self._mock_mode = (self.api_key == "your-api-key")
    
    async def _call_api(self, prompt: str, temperature: float = 0.7) -> str:
        """调用 API"""
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
                    "max_tokens": 1000,
                    "user": trace_id
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API 错误: {response.status_code}")
    
    def generate_variations(self, original_text: str, num_variations: int = 3) -> List[str]:
        """生成文本变体（数据增强）"""
        
        if self._mock_mode:
            return [original_text] * num_variations
        
        prompt = f"""请将以下文本改写成 {num_variations} 个不同的版本，保持原意但使用不同的表述方式。
每行一个版本，不要编号。

原文：
{original_text}

改写版本："""
        
        try:
            response = asyncio.run(self._call_api(prompt, temperature=0.8))
            variations = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return variations[:num_variations]
        except Exception as e:
            print(f"生成失败: {e}")
            return [original_text]
    
    def generate_decision_reason(self, case_info: dict) -> str:
        """生成决策理由"""
        
        prompt = f"""作为信贷审批专家，请根据以下客户信息，写一段专业的审批决策理由（50-100字）。

客户信息：
• 月收入：{case_info['monthly_income']} 元
• 负债比：{case_info['debt_ratio']}
• 逾期次数：{case_info['local_overdue']}
• 征信摘要：{case_info['credit_summary']}

已知决策：{case_info['decision']}
已知风险等级：{case_info['risk_level']}

请写出决策理由："""
        
        if self._mock_mode:
            return f"根据客户综合情况分析，{case_info['decision']}该申请。"
        
        try:
            return asyncio.run(self._call_api(prompt, temperature=0.3))
        except Exception as e:
            print(f"生成失败: {e}")
            return "风险综合评估后给出决策。"
    
    def generate_case_from_template(self, risk_level: str) -> dict:
        """根据风险等级生成新案例"""
        
        prompt = f"""请生成一个信贷审批案例，风险等级为「{risk_level}」。

请按以下 JSON 格式输出：
{{
  "monthly_income": 月收入（数字）,
  "debt_ratio": 负债比（0-1之间的小数）,
  "local_overdue": 本行逾期次数（数字）,
  "employment_years": 工作年限（数字）,
  "credit_summary": "征信报告摘要（一段描述，包含查询记录、担保情况、逾期历史等）",
  "risk_level": "{risk_level}",
  "decision": "审批建议（通过/拒绝/人工审核）",
  "reason": "决策理由",
  "credit_score": 信用评分（0-1000）
}}

只输出 JSON，不要其他内容。"""
        
        if self._mock_mode:
            return {
                "monthly_income": 20000,
                "debt_ratio": 0.3,
                "local_overdue": 0,
                "employment_years": 5,
                "credit_summary": "模拟案例",
                "risk_level": risk_level,
                "decision": "人工审核",
                "reason": "模拟决策理由",
                "credit_score": 600
            }
        
        try:
            response = asyncio.run(self._call_api(prompt, temperature=0.7))
            
            # 提取 JSON
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]
            
            return json.loads(response)
        except Exception as e:
            print(f"生成失败: {e}")
            return None
    
    def batch_generate(self, risk_levels: List[str], num_per_level: int = 5) -> List[dict]:
        """批量生成案例"""
        cases = []
        for level in risk_levels:
            print(f"正在生成 {level} 风险案例...")
            for i in range(num_per_level):
                case = self.generate_case_from_template(level)
                if case:
                    cases.append(case)
                    print(f"  ✓ 已生成 {len(cases)} 条")
        return cases


# ==========================================
# 主函数
# ==========================================

def main():
    print("="*60)
    print("🤖 LLM 辅助 SFT 数据生成")
    print("="*60)
    
    generator = LLMDataGenerator()
    
    if generator._mock_mode:
        print("\n⚠️ 未配置 API Key，将使用模拟模式")
        print("   设置方法: export LLM_API_KEY='你的AppID'")
    
    print("\n" + "-"*40)
    
    # 示例 1：生成决策理由
    print("\n📝 示例 1：生成决策理由")
    case_info = {
        "monthly_income": 25000,
        "debt_ratio": 0.3,
        "local_overdue": 0,
        "credit_summary": "近3个月有18次贷款查询，有50万担保",
        "decision": "拒绝",
        "risk_level": "高"
    }
    reason = generator.generate_decision_reason(case_info)
    print(f"生成的决策理由: {reason}")
    
    # 示例 2：批量生成案例
    print("\n📝 示例 2：批量生成案例")
    cases = generator.batch_generate(
        risk_levels=["低", "中", "高"],
        num_per_level=2
    )
    
    if cases:
        print(f"\n✅ 共生成 {len(cases)} 条案例")
        
        # 保存结果
        with open("generated_cases.json", "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print("💾 已保存到 generated_cases.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
