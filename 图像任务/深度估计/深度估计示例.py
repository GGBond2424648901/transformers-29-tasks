#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度估计实战示例
使用 Depth Pro 进行深度估计
"""

import os

# 设置模型缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("🌊 深度估计实战示例")
print("=" * 70)
print(f"📁 模型缓存路径: {os.environ['HF_HOME']}")
print("=" * 70)

# 1. 创建深度估计 pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)

depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("✅ 模型加载成功！")
print(f"   模型: depth-anything/Depth-Anything-V2-Small-hf")
print(f"   任务: 深度估计")

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
    print("   请使用本地图像文件")
    image = None

# 3. 进行深度估计
if image:
    print("\n🔍 步骤 3: 进行深度估计")
    print("-" * 70)
    
    result = depth_estimator(image)
    
    print("✅ 深度估计完成！")
    print(f"\n📊 结果信息:")
    print(f"   深度图类型: {type(result['depth'])}")
    print(f"   深度图大小: {result['depth'].size}")
    
    # 4. 可视化结果
    print("\n📊 步骤 4: 可视化深度图")
    print("-" * 70)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示原图
    axes[0].imshow(image)
    axes[0].set_title("原始图像", fontsize=14)
    axes[0].axis('off')
    
    # 显示深度图
    depth_map = np.array(result['depth'])
    im = axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title("深度图（暖色=近，冷色=远）", fontsize=14)
    axes[1].axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存结果
    output_path = "深度估计结果.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存到: {output_path}")
    
    # 显示图形
    try:
        plt.show()
    except:
        print("   (无法显示图形，但已保存到文件)")

# 5. 使用本地图像
print("\n" + "=" * 70)
print("💡 使用本地图像")
print("=" * 70)
print("""
如果要使用本地图像，可以这样做：

from PIL import Image

# 加载本地图像
image = Image.open("your_image.jpg")

# 深度估计
result = depth_estimator(image)

# 获取深度图
depth_map = result['depth']

# 保存深度图
depth_map.save("depth_map.png")
""")

# 6. 应用场景说明
print("\n" + "=" * 70)
print("🎯 应用场景")
print("=" * 70)
print("""
深度估计的主要应用：

1. 🤖 机器人导航
   - 障碍物检测
   - 路径规划
   - 避障系统

2. 🎮 AR/VR
   - 虚拟物体放置
   - 场景重建
   - 交互增强

3. 📷 摄影后期
   - 背景虚化
   - 景深效果
   - 3D 照片

4. 🚗 自动驾驶
   - 距离测量
   - 场景理解
   - 安全预警

5. 🏗️ 3D 重建
   - 建筑测量
   - 地形建模
   - 文物保护
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
