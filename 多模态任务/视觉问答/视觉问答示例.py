#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉问答（VQA）示例
根据图像回答问题
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("👁️💬 视觉问答示例")
print("=" * 70)

# 创建 VQA pipeline
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

print("✅ 模型加载成功！")

# 加载图像
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(BytesIO(requests.get(image_url).content))

# 提问
questions = [
    "What animal is in the image?",
    "What color is the cat?",
    "Is the cat sitting or standing?",
    "Where is the cat?"
]

print("\n📸 图像已加载")
print("\n🤔 开始提问：\n")

for question in questions:
    result = vqa(image=image, question=question)
    print(f"Q: {question}")
    print(f"A: {result[0]['answer']} (置信度: {result[0]['score']:.2%})\n")

print("""
应用场景：
- 🛍️ 智能客服 - 商品咨询
- 🔍 图片搜索 - 内容理解
- ♿ 无障碍辅助 - 图像描述
- 📚 教育 - 图像问答

使用技巧：
1. 问题要具体明确
2. 避免需要推理的复杂问题
3. 支持多种语言（需要对应模型）

推荐模型：
- Salesforce/blip-vqa-base: 通用 VQA
- Salesforce/blip2-opt-2.7b: 更强大的模型
- dandelin/vilt-b32-finetuned-vqa: ViLT 架构
""")

print("\n✨ 示例完成！")
