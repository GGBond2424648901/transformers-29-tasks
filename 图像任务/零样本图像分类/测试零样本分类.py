#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试零样本图像分类的输出格式
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("测试零样本图像分类")
print("=" * 70)

print("\n加载模型...")
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=0)
print("模型加载完成！")

# 下载一张测试图片
print("\n下载测试图片...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print("图片下载完成！")

# 测试分类
labels = ["猫", "狗", "鸟", "汽车", "飞机"]
print(f"\n测试标签: {labels}")

print("\n开始分类...")
results = classifier(image, candidate_labels=labels)

print("\n结果类型:", type(results))
print("结果内容:")
print(results)

# 尝试解析结果
print("\n" + "=" * 70)
print("解析结果:")
print("=" * 70)

if isinstance(results, list):
    print("返回格式: 列表")
    for i, item in enumerate(results):
        print(f"{i+1}. {item}")
elif isinstance(results, dict):
    print("返回格式: 字典")
    for key, value in results.items():
        print(f"{key}: {value}")

print("\n测试完成！")
