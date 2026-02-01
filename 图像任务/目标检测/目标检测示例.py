#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测实战示例
使用 DETR (Detection Transformer) 进行目标检测
"""

from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

print("=" * 70)
print("🎯 目标检测实战示例")
print("=" * 70)

# 1. 创建目标检测 pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)

detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"
)

print("✅ 模型加载成功！")
print(f"   模型: facebook/detr-resnet-50")
print(f"   任务: 目标检测")

# 2. 准备测试图像
print("\n🖼️  步骤 2: 准备测试图像")
print("-" * 70)

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

try:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    print(f"✅ 从 URL 加载图像成功")
    print(f"   图像大小: {image.size}")
except Exception as e:
    print(f"⚠️  无法从 URL 加载图像: {e}")
    image = None

# 3. 进行目标检测
if image:
    print("\n🔍 步骤 3: 进行目标检测")
    print("-" * 70)
    
    results = detector(image)
    
    print(f"✅ 检测完成！检测到 {len(results)} 个对象")
    print("\n📊 检测结果:")
    for i, result in enumerate(results, 1):
        print(f"\n   对象 {i}:")
        print(f"      类别: {result['label']}")
        print(f"      置信度: {result['score']:.2%}")
        print(f"      位置: {result['box']}")

# 4. 可视化结果
if image and results:
    print("\n🎨 步骤 4: 可视化检测结果")
    print("-" * 70)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 为每个检测到的对象绘制边界框
    for result in results:
        box = result['box']
        label = result['label']
        score = result['score']
        
        # 提取坐标
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']
        
        # 绘制矩形框
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline="red",
            width=3
        )
        
        # 添加标签
        text = f"{label} {score:.2f}"
        draw.text((xmin, ymin - 10), text, fill="red")
    
    # 保存结果
    output_path = "detection_result.jpg"
    image.save(output_path)
    print(f"✅ 可视化结果已保存到: {output_path}")

# 5. 设置检测阈值
print("\n⚙️  步骤 5: 调整检测参数")
print("-" * 70)

# 只保留置信度 > 0.9 的检测结果
detector_high_threshold = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"
)

if image:
    results_filtered = detector_high_threshold(image, threshold=0.9)
    print(f"✅ 高置信度检测: 检测到 {len(results_filtered)} 个对象")
    for result in results_filtered:
        print(f"   - {result['label']}: {result['score']:.2%}")

# 6. 使用说明
print("\n" + "=" * 70)
print("💡 使用本地图像")
print("=" * 70)
print("""
如果要使用本地图像，可以这样做：

from PIL import Image

# 加载本地图像
image = Image.open("your_image.jpg")

# 检测对象
results = detector(image)

# 查看结果
for result in results:
    print(f"检测到: {result['label']}")
    print(f"位置: {result['box']}")
    print(f"置信度: {result['score']:.2%}")
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
