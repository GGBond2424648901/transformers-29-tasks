#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档理解示例
OCR + 文档结构理解
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("📄🔍 文档理解示例")
print("=" * 70)

# 创建文档问答 pipeline
doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

print("✅ 模型加载成功！")

print("""
应用场景：
- 📋 发票识别与信息提取
- 📑 合同分析与关键信息抽取
- 📊 表格数据提取
- 🏦 金融文档处理
- 📝 表单自动填充

功能特点：
1. OCR - 文字识别
2. 布局分析 - 理解文档结构
3. 信息提取 - 回答关于文档的问题

使用方法：
```python
# 对文档提问
result = doc_qa(
    image="invoice.png",
    question="What is the total amount?"
)
print(result)
# [{'answer': '$1,234.56', 'score': 0.95}]

# 提取多个字段
questions = [
    "What is the invoice number?",
    "What is the date?",
    "Who is the vendor?"
]

for question in questions:
    result = doc_qa(image="invoice.png", question=question)
    print(f"{question}: {result[0]['answer']}")
```

推荐模型：
- impira/layoutlm-document-qa: 通用文档问答
- microsoft/layoutlmv3-base: 更强大的文档理解
- naver-clova-ix/donut-base: 端到端文档理解

相关任务：
- 表格问答 (Table QA)
- 视觉问答 (VQA)
- OCR (Optical Character Recognition)
""")

print("\n✨ 示例完成！")
