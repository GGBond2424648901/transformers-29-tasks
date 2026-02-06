#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键点检测与匹配示例
使用 SuperPoint 和 SuperGlue 进行图像特征匹配
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

print("=" * 70)
print("🔍 关键点检测与匹配示例")
print("=" * 70)

print("""
⚠️  注意：
关键点检测任务需要使用专门的库，如 OpenCV、Kornia 等。

推荐工具：
1. SuperPoint + SuperGlue (最先进)
2. SIFT / SURF (经典算法)
3. ORB (快速)

应用场景：
- 📷 图像拼接、全景图
- 🏗️ 3D 重建
- 🤖 SLAM (同时定位与地图构建)
- 🔍 物体识别与追踪

安装：
pip install opencv-python kornia

示例代码：
```python
import cv2

# 使用 SIFT 检测关键点
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# 使用 BFMatcher 匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# 筛选好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```
""")

print("✨ 查看 README.md 了解更多信息")
