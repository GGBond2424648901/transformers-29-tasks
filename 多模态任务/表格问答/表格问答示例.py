#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格问答示例 - 命令行版本
绕过Web服务，直接使用transformers pipeline
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
import pandas as pd

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("📊 表格问答示例 - 命令行版本")
print("=" * 70)

# 加载模型
print("\n📚 正在加载表格问答模型...")
table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
print("✅ 模型加载完成！")

# 示例1：英文表格（推荐）
print("\n" + "=" * 70)
print("示例1：销售数据表格（英文）")
print("=" * 70)

# 读取CSV文件 - 使用绝对路径
csv_path = os.path.join(CURRENT_DIR, "销售数据测试.csv")
df_sales = pd.read_csv(csv_path)
print("\n表格内容：")
print(df_sales)

# 提问
questions = [
    "How many laptops were sold?",
    "What is the total revenue?",
    "Which product sold the most in January?"
]

for question in questions:
    print(f"\n❓ 问题: {question}")
    try:
        result = table_qa(table=df_sales, query=question)
        print(f"💡 答案: {result['answer']}")
        if 'coordinates' in result:
            print(f"📍 位置: {result['coordinates']}")
        if 'cells' in result:
            print(f"📋 相关单元格: {result['cells']}")
    except Exception as e:
        print(f"❌ 错误: {e}")

# 示例2：简单的员工表格
print("\n" + "=" * 70)
print("示例2：员工信息表格（简化版）")
print("=" * 70)

# 创建简单的DataFrame（避免复杂的中文处理）
data = {
    "Name": ["Zhang San", "Li Si", "Wang Wu", "Zhao Liu"],
    "Age": ["25", "30", "28", "35"],
    "Department": ["Tech", "Sales", "Tech", "Management"],
    "Salary": ["8000", "9000", "8500", "12000"]
}

df_employees = pd.DataFrame(data)
print("\n表格内容：")
print(df_employees)

# 提问
questions = [
    "How many people work in Tech?",
    "What is the average salary?",
    "Who has the highest salary?"
]

for question in questions:
    print(f"\n❓ 问题: {question}")
    try:
        result = table_qa(table=df_employees, query=question)
        print(f"💡 答案: {result['answer']}")
    except Exception as e:
        print(f"❌ 错误: {e}")

print("\n" + "=" * 70)
print("✅ 示例运行完成！")
print("=" * 70)
print("\n💡 提示：")
print("1. TAPAS模型主要为英文表格设计，英文问题效果最好")
print("2. 表格数据建议使用英文，避免编码问题")
print("3. 问题要具体明确，避免过于复杂的查询")
print("4. 如需处理中文表格，建议先翻译成英文")
