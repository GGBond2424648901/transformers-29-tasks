# 🌊 深度估计（Depth Estimation）

## 📖 任务简介

深度估计是计算机视觉中的一项重要任务，目标是从单张 2D 图像中估计场景的深度信息，即图像中每个像素点到相机的距离。

## 🎯 应用场景

- **🤖 机器人导航**: 障碍物检测、路径规划
- **🎮 AR/VR**: 虚拟物体放置、场景重建
- **📷 摄影后期**: 背景虚化、景深效果
- **🚗 自动驾驶**: 距离测量、场景理解
- **🏗️ 3D 重建**: 建筑测量、地形建模

## 📁 文件说明

- `深度估计示例.py` - Pipeline 推理示例（推荐入门）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install transformers pillow requests matplotlib numpy
```

### 2. 运行示例

```bash
python 深度估计示例.py
```

## 💡 使用示例

```python
from transformers import pipeline
from PIL import Image

# 创建深度估计器
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

# 加载图像
image = Image.open("photo.jpg")

# 估计深度
result = depth_estimator(image)

# 获取深度图
depth_map = result['depth']

# 保存深度图
depth_map.save("depth_map.png")
```

## 🎨 推荐模型

| 模型 | 大小 | 特点 |
|------|------|------|
| depth-anything/Depth-Anything-V2-Small-hf | 小 | 快速，适合实时应用 |
| depth-anything/Depth-Anything-V2-Base-hf | 中 | 平衡性能和速度 |
| depth-anything/Depth-Anything-V2-Large-hf | 大 | 最高精度 |
| Intel/dpt-large | 大 | 经典模型，效果稳定 |

## 📊 输出说明

深度估计的输出包含：
- `depth`: PIL Image 对象，表示深度图
- 深度图中，亮度表示距离（通常暖色=近，冷色=远）

## ⚠️ 注意事项

1. **模型大小**: 深度估计模型通常较大，首次下载需要时间
2. **GPU 加速**: 建议使用 GPU 以获得更快的推理速度
3. **图像质量**: 输入图像质量会影响深度估计的准确性
4. **相对深度**: 大多数模型估计的是相对深度，而非绝对距离

## 🔗 相关资源

- [Depth Anything 论文](https://arxiv.org/abs/2401.10891)
- [Hugging Face 深度估计文档](https://huggingface.co/docs/transformers/tasks/depth_estimation)
