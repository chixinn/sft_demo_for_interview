# -*- coding: utf-8 -*-
"""
SFT 数据集构建工具
用于将真实业务数据转换为模型训练格式
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import csv

# ==========================================
# 1. 数据结构定义
# ==========================================

@dataclass
class CreditCase:
    """信贷案例数据结构"""
    case_id: str
    # 结构化特征
    monthly_income: float
    debt_ratio: float
    local_overdue: int
    employment_years: int
    # 非结构化特征（征信摘要）
    credit_report_summary: str
    # 标签（专家决策）
    risk_level: str  # 低/中/高/极高
    decision: str    # 通过/拒绝/人工审核
    reason: str      # 决策理由
    credit_score: int  # 信用评分 0-1000


@dataclass
class SFTSample:
    """SFT 训练样本"""
    instruction: str
    input: str
    output: str


# ==========================================
# 2. 模板设计
# ==========================================

INSTRUCTION_TEMPLATES = [
    "作为信贷审批专家，请分析以下客户的风险特征并给出审批建议。",
    "你是一个专业的信贷风控系统，请根据客户信息进行风险评估。",
    "基于以下客户数据，请给出信贷审批决策和理由。",
    "请分析客户信用状况，输出风险等级、审批建议和决策理由。",
]

INPUT_TEMPLATE = """【客户基本信息】
• 月收入：{monthly_income} 元
• 负债比：{debt_ratio}
• 本行逾期次数：{local_overdue}
• 工作年限：{employment_years} 年

【征信报告摘要】
{credit_report_summary}"""

OUTPUT_TEMPLATE = """风险等级：{risk_level}
信用评分：{credit_score}
审批建议：{decision}
决策理由：{reason}"""


# ==========================================
# 3. 数据构建器
# ==========================================

class SFTDatasetBuilder:
    """SFT 数据集构建器"""
    
    def __init__(self):
        self.samples: List[SFTSample] = []
    
    def add_case(self, case: CreditCase):
        """添加单个案例"""
        # 随机选择指令模板，增加多样性
        instruction = random.choice(INSTRUCTION_TEMPLATES)
        
        # 构建输入
        input_text = INPUT_TEMPLATE.format(
            monthly_income=case.monthly_income,
            debt_ratio=case.debt_ratio,
            local_overdue=case.local_overdue,
            employment_years=case.employment_years,
            credit_report_summary=case.credit_report_summary
        )
        
        # 构建输出
        output_text = OUTPUT_TEMPLATE.format(
            risk_level=case.risk_level,
            credit_score=case.credit_score,
            decision=case.decision,
            reason=case.reason
        )
        
        self.samples.append(SFTSample(
            instruction=instruction,
            input=input_text,
            output=output_text
        ))
    
    def from_csv(self, csv_path: str):
        """从 CSV 文件加载数据"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = CreditCase(
                    case_id=row['case_id'],
                    monthly_income=float(row['monthly_income']),
                    debt_ratio=float(row['debt_ratio']),
                    local_overdue=int(row['local_overdue']),
                    employment_years=int(row['employment_years']),
                    credit_report_summary=row['credit_report_summary'],
                    risk_level=row['risk_level'],
                    decision=row['decision'],
                    reason=row['reason'],
                    credit_score=int(row['credit_score'])
                )
                self.add_case(case)
        print(f"✅ 从 CSV 加载了 {len(self.samples)} 条数据")
    
    def add_synthetic_samples(self, base_case: CreditCase, num_variations: int = 5):
        """基于现有案例生成变体（数据增强）"""
        for i in range(num_variations):
            # 随机调整数值特征
            variation = CreditCase(
                case_id=f"{base_case.case_id}_var_{i}",
                monthly_income=base_case.monthly_income * random.uniform(0.8, 1.2),
                debt_ratio=min(1.0, base_case.debt_ratio * random.uniform(0.8, 1.2)),
                local_overdue=max(0, base_case.local_overdue + random.randint(-1, 1)),
                employment_years=max(0, base_case.employment_years + random.randint(-2, 2)),
                credit_report_summary=base_case.credit_report_summary,
                risk_level=base_case.risk_level,
                decision=base_case.decision,
                reason=base_case.reason,
                credit_score=max(0, min(1000, base_case.credit_score + random.randint(-50, 50)))
            )
            self.add_case(variation)
    
    def split_dataset(self, train_ratio: float = 0.8) -> tuple:
        """划分训练集和验证集"""
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        train_set = self.samples[:split_idx]
        val_set = self.samples[split_idx:]
        return train_set, val_set
    
    def save_alpaca_format(self, output_path: str):
        """保存为 Alpaca 格式"""
        data = [asdict(sample) for sample in self.samples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存 {len(data)} 条数据到 {output_path}")
    
    def save_sharegpt_format(self, output_path: str):
        """保存为 ShareGPT 格式"""
        data = []
        for sample in self.samples:
            data.append({
                "conversations": [
                    {"from": "human", "value": f"{sample.instruction}\n\n{sample.input}"},
                    {"from": "assistant", "value": sample.output}
                ]
            })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存 {len(data)} 条数据到 {output_path}")


# ==========================================
# 4. 示例：构建示例数据集
# ==========================================

def create_sample_dataset():
    """创建示例数据集"""
    builder = SFTDatasetBuilder()
    
    # 示例案例 1：高风险客户
    case1 = CreditCase(
        case_id="APP-2023-0001",
        monthly_income=25000,
        debt_ratio=0.3,
        local_overdue=0,
        employment_years=5,
        credit_report_summary="""近3个月内有18次贷款审批查询记录。
客户作为保证人，为关联企业提供50万担保，该企业近期被法院列为被执行人。
历史逾期记录：2年前有一次30天逾期。""",
        risk_level="高",
        decision="拒绝",
        reason="存在多头借贷倾向，且有高风险担保债务，综合判定违约概率较高。",
        credit_score=420
    )
    
    # 示例案例 2：低风险客户
    case2 = CreditCase(
        case_id="APP-2023-0002",
        monthly_income=35000,
        debt_ratio=0.2,
        local_overdue=0,
        employment_years=8,
        credit_report_summary="""无近期贷款查询记录。
无担保记录。
历史信用良好，无逾期记录。
名下有房产一套，车辆一辆。""",
        risk_level="低",
        decision="通过",
        reason="收入稳定，负债比例健康，信用记录良好，风险可控。",
        credit_score=850
    )
    
    # 示例案例 3：中等风险客户
    case3 = CreditCase(
        case_id="APP-2023-0003",
        monthly_income=18000,
        debt_ratio=0.45,
        local_overdue=0,
        employment_years=3,
        credit_report_summary="""近6个月内有3次贷款查询记录。
无担保记录。
历史有一次60天逾期，发生在1年前。
信用卡使用率较高，约85%。""",
        risk_level="中",
        decision="人工审核",
        reason="负债比例偏高，信用卡使用率高，历史有逾期记录，建议人工核实还款能力。",
        credit_score=580
    )
    
    # 添加案例
    builder.add_case(case1)
    builder.add_case(case2)
    builder.add_case(case3)
    
    # 数据增强：为每个案例生成变体
    builder.add_synthetic_samples(case1, num_variations=10)
    builder.add_synthetic_samples(case2, num_variations=10)
    builder.add_synthetic_samples(case3, num_variations=10)
    
    return builder


def main():
    print("="*60)
    print("🔧 SFT 数据集构建工具")
    print("="*60)
    
    # 创建示例数据集
    builder = create_sample_dataset()
    
    print(f"\n📊 数据集统计:")
    print(f"   • 总样本数: {len(builder.samples)}")
    
    # 划分训练集和验证集
    train_set, val_set = builder.split_dataset(train_ratio=0.9)
    print(f"   • 训练集: {len(train_set)} 条")
    print(f"   • 验证集: {len(val_set)} 条")
    
    # 保存数据集
    print("\n💾 保存数据集...")
    builder.save_alpaca_format("sft_data_alpaca.json")
    builder.save_sharegpt_format("sft_data_sharegpt.json")
    
    # 显示示例
    print("\n📝 示例数据:")
    print("-"*40)
    sample = builder.samples[0]
    print(f"Instruction: {sample.instruction}")
    print(f"\nInput:\n{sample.input}")
    print(f"\nOutput:\n{sample.output}")
    print("="*60)


if __name__ == "__main__":
    main()
