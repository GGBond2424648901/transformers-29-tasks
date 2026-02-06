#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

print("=" * 70)
print("🔍 测试模型加载")
print("=" * 70)

print("\n环境变量:")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

print("\n正在加载模型...")
print("模型: impira/layoutlm-document-qa")
print("这可能需要几分钟时间（首次下载）...")

try:
    from transformers import pipeline
    doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
    print("\n✅ 模型加载成功！")
    print("文档理解服务可以正常使用了。")
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    print("\n可能的原因:")
    print("1. 网络连接问题")
    print("2. 镜像站点访问问题")
    print("3. 模型文件损坏")
    print("\n建议:")
    print("1. 检查网络连接")
    print("2. 稍后重试")
    print("3. 或使用其他镜像站点")

print("\n" + "=" * 70)
