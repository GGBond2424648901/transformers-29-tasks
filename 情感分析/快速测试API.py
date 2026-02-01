#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 API - 简化版
"""

import requests
import json

API_URL = "http://localhost:5000"

print("=" * 60)
print("测试情感分析 API")
print("=" * 60)

# 测试 1: 健康检查
print("\n1️⃣ 健康检查...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"✓ 状态: {response.json()}")
except Exception as e:
    print(f"✗ 错误: {e}")

# 测试 2: 单个预测
print("\n2️⃣ 单个文本预测...")
test_texts = [
    "这个产品质量很好！",
    "太差了，不推荐",
    "还可以，一般般"
]

for text in test_texts:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "use_pipeline": False}
        )
        result = response.json()
        print(f"\n文本: {text}")
        print(f"情感: {result['sentiment']}")
        print(f"置信度: {result['confidence']:.2%}")
    except Exception as e:
        print(f"✗ 错误: {e}")

# 测试 3: 批量预测
print("\n3️⃣ 批量预测...")
batch_texts = [
    "物流很快，推荐购买",
    "客服态度差",
    "性价比不错"
]

try:
    response = requests.post(
        f"{API_URL}/batch_predict",
        json={"texts": batch_texts}
    )
    result = response.json()
    print(f"\n共 {result['count']} 条结果:")
    for item in result['results']:
        print(f"  • {item['text']} → {item['sentiment']} ({item['score']:.2%})")
except Exception as e:
    print(f"✗ 错误: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
