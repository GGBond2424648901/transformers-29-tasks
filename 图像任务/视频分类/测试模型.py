#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试VideoMAE fine-tuned模型
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("测试VideoMAE fine-tuned模型")
print("=" * 70)

print("\n加载模型...")
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("模型加载完成！")

# 下载测试图片
print("\n下载测试图片...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
print("图片下载完成！")

# 复制为16帧
frames = [image] * 16
print(f"\n图片已复制为{len(frames)}帧")

# 进行分类
print("\n开始分类...")
inputs = processor(frames, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

# 获取top-5结果
num_classes = logits.shape[-1]
k = min(5, num_classes)
top_probs, top_indices = torch.topk(probs, k)

print(f"\n模型类别总数: {num_classes}")
print(f"\nTop-{k} 预测结果:")
print("=" * 70)

for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
    label = model.config.id2label.get(idx.item(), f"类别_{idx.item()}")
    print(f"{i+1}. {label}: {prob.item()*100:.2f}%")

print("\n" + "=" * 70)
print("测试完成！模型工作正常。")
