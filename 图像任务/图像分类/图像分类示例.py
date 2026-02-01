#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类实战示例
使用 ViT (Vision Transformer) 进行图像分类
"""

import os

# 设置模型缓存路径（在导入 transformers 之前）
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("🖼️  图像分类实战示例")
print("=" * 70)
print(f"📁 模型缓存路径: {os.environ['HF_HOME']}")
print("=" * 70)

# 1. 创建图像分类 pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

print("✅ 模型加载成功！")
print(f"   模型: google/vit-base-patch16-224")
print(f"   任务: 图像分类")

# 2. 准备测试图像
print("\n🖼️  步骤 2: 准备测试图像")
print("-" * 70)

# 方法 A: 从 URL 加载图像
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

try:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    print(f"✅ 从 URL 加载图像成功")
    print(f"   图像大小: {image.size}")
except Exception as e:
    print(f"⚠️  无法从 URL 加载图像: {e}")
    print("   请使用本地图像文件")
    image = None

# 方法 B: 从本地文件加载（如果有）
# image = Image.open("cat.jpg")

# 3. 进行分类
if image:
    print("\n🔍 步骤 3: 进行图像分类")
    print("-" * 70)
    
    results = classifier(image)
    
    print("✅ 分类完成！")
    print("\n📊 分类结果（Top 5）:")
    for i, result in enumerate(results[:5], 1):
        print(f"   {i}. {result['label']:<30} 置信度: {result['score']:.2%}")

# 4. 批量分类
print("\n📦 步骤 4: 批量图像分类")
print("-" * 70)

# 准备多张图像
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
]

images = []
for url in image_urls:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        images.append(img)
    except:
        pass

if images:
    print(f"✅ 加载了 {len(images)} 张图像")
    
    # 批量分类
    batch_results = classifier(images)
    
    print("\n📊 批量分类结果:")
    for i, results in enumerate(batch_results, 1):
        print(f"\n   图像 {i}:")
        for j, result in enumerate(results[:3], 1):
            print(f"      {j}. {result['label']:<25} {result['score']:.2%}")

# 5. 使用本地图像
print("\n" + "=" * 70)
print("💡 使用本地图像")
print("=" * 70)
print("""
如果要使用本地图像，可以这样做：

from PIL import Image

# 加载本地图像
image = Image.open("your_image.jpg")

# 分类
results = classifier(image)

# 查看结果
for result in results:
    print(f"{result['label']}: {result['score']:.2%}")
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
