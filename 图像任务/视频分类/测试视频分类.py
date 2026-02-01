#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频分类功能
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
from PIL import Image
import numpy as np

print("=" * 70)
print("🎬 测试视频分类模型")
print("=" * 70)

print("\n📦 加载模型...")
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ 模型加载完成，使用设备: {device}")

# 创建测试图像（16帧）- 使用numpy数组
print("\n🖼️ 创建测试图像...")
test_frames = []
for i in range(16):
    # 创建一个渐变的测试图像，使用numpy数组
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = i * 15  # R通道
    img_array[:, :, 1] = 100      # G通道
    img_array[:, :, 2] = 200      # B通道
    img = Image.fromarray(img_array)
    test_frames.append(img)

print(f"✅ 创建了 {len(test_frames)} 帧测试图像，每帧大小: {test_frames[0].size}")

# 测试处理器
print("\n🔧 测试图像处理器...")
try:
    # VideoMAE处理器期望输入是 [video1, video2, ...] 格式
    # 每个video是一个包含多帧的列表
    # 我们只有一个视频，所以包装成 [frames]
    inputs = processor(
        test_frames,  # 直接传递帧列表
        return_tensors="pt"
    )
    print(f"✅ 处理器输出: {inputs.keys()}")
    print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
    
    # 移到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 测试模型推理
    print("\n🤖 测试模型推理...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    print(f"✅ 模型输出 logits shape: {logits.shape}")
    print(f"   - 类别数量: {logits.shape[-1]}")
    
    # 获取top-5结果
    top_probs, top_indices = torch.topk(probs, 5)
    
    print("\n🏆 Top-5 预测结果:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        label = model.config.id2label.get(idx.item(), f"类别_{idx.item()}")
        print(f"   {i}. {label}: {prob.item()*100:.2f}%")
    
    print("\n✅ 所有测试通过！视频分类功能正常工作")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
