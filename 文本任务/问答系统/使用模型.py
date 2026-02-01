#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的问答模型
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("❓ 问答系统推理示例")
print("=" * 70)

# 加载模型
print("\n📦 加载模型...")

# 使用训练好的模型
model_path = "./my_qa_model"

# 如果还没训练，使用预训练模型
if not os.path.exists(model_path):
    print("⚠️  未找到训练好的模型，使用预训练模型")
    model_path = "bert-base-uncased"

qa_pipeline = pipeline(
    "question-answering",
    model=model_path
)

print(f"✅ 模型加载成功: {model_path}")

# 测试问答
print("\n" + "=" * 70)
print("🧪 测试问答")
print("=" * 70)

# 示例 1
context = """
Transformers is a library maintained by Hugging Face. It provides 
thousands of pretrained models to perform tasks on different modalities 
such as text, vision, and audio. The library is designed to be easy to 
use and allows researchers and developers to quickly experiment with 
state-of-the-art models.
"""

questions = [
    "Who maintains Transformers?",
    "What does Transformers provide?",
    "What modalities does it support?"
]

print(f"\n📄 上下文:\n{context.strip()}\n")

for i, question in enumerate(questions, 1):
    result = qa_pipeline(question=question, context=context)
    
    print(f"\n{i}. 问题: {question}")
    print(f"   答案: {result['answer']}")
    print(f"   置信度: {result['score']:.2%}")

# 示例 2 - 中文（如果使用中文模型）
print("\n" + "=" * 70)
print("💡 使用技巧")
print("=" * 70)

print("""
1. 上下文要包含答案
2. 问题要清晰具体
3. 模型会返回最可能的答案片段

使用方法：
```python
result = qa_pipeline(
    question="你的问题",
    context="包含答案的上下文"
)

print(result['answer'])  # 答案
print(result['score'])   # 置信度
```

训练中文模型：
1. 使用中文预训练模型（如 bert-base-chinese）
2. 使用中文问答数据集（如 CMRC2018）
3. 修改 简单训练示例.py 中的 model_name
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
