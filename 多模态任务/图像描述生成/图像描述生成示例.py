#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像描述生成示例
自动为图像生成文字描述
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("📸💬 图像描述生成示例")
print("=" * 70)

# 创建图像描述生成器
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

print("✅ 模型加载成功！")

# 加载图像
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(BytesIO(requests.get(image_url).content))

print("\n📸 生成图像描述...")

# 生成描述
result = captioner(image)
print(f"\n描述: {result[0]['generated_text']}")

# 生成多个描述
print("\n📝 生成多个候选描述：")
results = captioner(image, max_new_tokens=50, num_beams=5, num_return_sequences=3)

for i, res in enumerate(results, 1):
    print(f"{i}. {res['generated_text']}")

print("""
\n应用场景：
- ♿ 无障碍辅助 - 为视障人士描述图像
- 🔍 图片 SEO - 自动生成 alt 文本
- 📱 社交媒体 - 自动生成图片说明
- 📚 内容管理 - 图片自动标注

使用技巧：
1. 调整 max_new_tokens 控制描述长度
2. 使用 num_beams 提高描述质量
3. num_return_sequences 生成多个候选

推荐模型：
- Salesforce/blip-image-captioning-base: 通用描述
- Salesforce/blip2-opt-2.7b: 更详细的描述
- nlpconnect/vit-gpt2-image-captioning: 轻量级
""")

print("\n✨ 示例完成！")
