#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试情感分析 API 的客户端
需要先运行 模型部署示例.py 启动服务
"""

import requests
import json

# API 地址
API_URL = "http://localhost:5000"

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("1. 测试健康检查")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_single_predict():
    """测试单个文本预测"""
    print("=" * 60)
    print("2. 测试单个文本预测")
    print("=" * 60)
    
    test_cases = [
        "这个产品质量很好，非常满意！",
        "太差了，完全不值这个价格",
        "还可以，一般般"
    ]
    
    for text in test_cases:
        data = {"text": text, "use_pipeline": False}
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n文本: {text}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    print()

def test_batch_predict():
    """测试批量预测"""
    print("=" * 60)
    print("3. 测试批量预测")
    print("=" * 60)
    
    texts = [
        "物流很快，包装完好，推荐购买",
        "客服态度恶劣，再也不买了",
        "性价比不错，值得入手",
        "质量一般，价格偏贵",
        "超出预期，五星好评！"
    ]
    
    data = {"texts": texts}
    response = requests.post(
        f"{API_URL}/batch_predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    result = response.json()
    print(f"总数: {result['count']}")
    print("\n结果:")
    for item in result['results']:
        print(f"  - {item['text']}")
        print(f"    情感: {item['sentiment']} (置信度: {item['score']:.2%})")
    
    print()

def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("4. 测试错误处理")
    print("=" * 60)
    
    # 测试缺少参数
    print("\n测试缺少 text 参数:")
    response = requests.post(
        f"{API_URL}/predict",
        json={},
        headers={"Content-Type": "application/json"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # 测试错误的数据类型
    print("\n测试错误的数据类型:")
    response = requests.post(
        f"{API_URL}/batch_predict",
        json={"texts": "not a list"},
        headers={"Content-Type": "application/json"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    print()

if __name__ == '__main__':
    try:
        print("\n" + "=" * 60)
        print("开始测试情感分析 API")
        print("=" * 60 + "\n")
        
        # 运行所有测试
        test_health()
        test_single_predict()
        test_batch_predict()
        test_error_handling()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到 API 服务")
        print("请先运行 模型部署示例.py 启动服务")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
