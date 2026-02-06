#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载中文文本转语音模型
"""

import os

# 设置缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

# 使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 70)
print("📥 下载中文文本转语音模型")
print("=" * 70)
print("\n使用镜像: https://hf-mirror.com")
print("模型: facebook/mms-tts-chz (支持中文)")
print("\n开始下载...\n")

try:
    from transformers import VitsModel, AutoTokenizer
    import torch
    
    # 下载模型和分词器
    print("📦 下载模型...")
    model = VitsModel.from_pretrained("facebook/mms-tts-chz")
    
    print("📦 下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-chz")
    
    print("\n" + "=" * 70)
    print("✅ 模型下载成功!")
    print("=" * 70)
    
    # 测试模型
    print("\n🧪 测试模型...")
    test_text = "你好，这是一个测试。"
    
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    print("✅ 模型测试成功!")
    print(f"   采样率: {model.config.sampling_rate} Hz")
    print(f"   音频形状: {output.shape}")
    print(f"   测试文本: {test_text}")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ 下载失败!")
    print("=" * 70)
    print(f"\n错误信息: {e}")
    print("\n💡 解决方案:")
    print("   1. 检查网络连接")
    print("   2. 确认镜像地址可访问")
    print("   3. 尝试使用 VPN")
    print("   4. 或手动从 https://hf-mirror.com/facebook/mms-tts-chz 下载")
