#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格问答示例
对结构化表格数据进行问答
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
import pandas as pd

print("=" * 70)
print("📊💬 表格问答示例")
print("=" * 70)

# 创建表格问答 pipeline
table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")

print("✅ 模型加载成功！")

# 创建示例表格
table = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 28],
    "部门": ["技术部", "销售部", "技术部"],
    "工资": [8000, 9000, 8500]
})

print("\n📊 示例表格：")
print(table)

# 提问
questions = [
    "技术部有多少人？",
    "谁的工资最高？",
    "平均工资是多少？"
]

print("\n🤔 开始提问：\n")

for question in questions:
    result = table_qa(table=table, query=question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")

print("""
应用场景：
- 📊 数据分析 - 自然语言查询数据
- 💼 财务报表 - 快速查找信息
- 📈 业务报告 - 智能问答
- 🏢 企业数据 - 员工信息查询

使用技巧：
1. 表格需要是 pandas DataFrame 格式
2. 问题要与表格内容相关
3. 支持聚合查询（求和、平均等）

推荐模型：
- google/tapas-base-finetuned-wtq: 通用表格问答
- microsoft/tapex-large: 更强大的模型
- neulab/omnitab-large: 支持复杂查询

示例代码：
```python
import pandas as pd

# 创建表格
table = pd.DataFrame({
    "产品": ["A", "B", "C"],
    "销量": [100, 200, 150],
    "价格": [10, 20, 15]
})

# 提问
result = table_qa(
    table=table,
    query="哪个产品销量最高？"
)
print(result['answer'])
```
""")

print("\n✨ 示例完成！")
