# 🤸 姿态估计（Pose Estimation）

## 📖 任务简介
姿态估计是检测图像或视频中人体关键点位置的任务，可以识别人体的姿势和动作。本项目提供**真正的人体骨骼关键点检测**，而不是简单的图像分类。

## 🎯 应用场景
- 🏃 运动分析、健身指导
- 🎮 体感游戏、动作捕捉
- 🏥 医疗康复、步态分析
- 🎬 动画制作、特效处理

## 🦴 检测的关键点（17个）
1. **头部**：鼻子、左眼、右眼、左耳、右耳
2. **上半身**：左肩、右肩、左肘、右肘、左腕、右腕
3. **下半身**：左髋、右髋、左膝、右膝、左踝、右踝

## 🚀 Web服务

### 启动服务
```bash
D:\aaaalokda\envs\myenv\python.exe "姿态估计Web服务.py"
```

### 访问地址
http://localhost:6005

### 功能特点
- ✅ 真正的骨骼关键点检测
- ✅ 双图对比（原图 + 骨骼图）
- ✅ 关键点列表显示
- ✅ 运动少女主题 + 能量波纹动画
- ✅ 自定义背景图片

## 📦 依赖安装

### 基础依赖（必需）
```bash
pip install flask pillow numpy
```

### OpenPose检测器（推荐）
```bash
pip install controlnet-aux
```

**说明**：
- 安装 `controlnet-aux` 后使用专业的OpenPose技术
- 未安装则使用简化版本绘制示例骨架

## 🛠️ 推荐工具
- **OpenPose**: 经典算法，多人姿态估计（本项目使用）
- **MediaPipe**: Google 开发，实时性能好
- **MMPose**: 功能全面，支持多种算法
- **ViTPose**: 基于Vision Transformer的姿态估计

## 💡 快速开始

### 方式1：Web服务（推荐）
```bash
# 启动Web服务
D:\aaaalokda\envs\myenv\python.exe "姿态估计Web服务.py"

# 浏览器访问
http://localhost:6005
```

### 方式2：命令行示例
```bash
pip install mediapipe opencv-python
python 姿态估计示例.py
```

## 🎨 界面预览

- **主题**：橙色运动少女风格
- **动画**：能量波纹飘落效果
- **布局**：左右对比（原图 + 骨骼图）
- **背景**：自定义背景图片

## 📝 使用说明

1. 上传包含人物的图片（建议全身照）
2. 点击"开始检测"按钮
3. 查看右侧的骨骼检测结果
4. 下方显示检测到的关键点列表

## ⚠️ 注意事项

1. **图片质量**：清晰的全身照片效果最好
2. **人物姿态**：正面或侧面姿态检测效果较好
3. **背景干扰**：简单背景有助于提高检测准确度
4. **多人检测**：当前版本主要针对单人检测

## 🆚 与图像分类的区别

| 特性 | 图像分类 | 姿态估计 |
|------|---------|---------|
| 输出 | 图片类别标签 | 人体关键点坐标 |
| 用途 | 识别图片内容 | 检测人体姿态 |
| 结果 | "跑步"、"跳跃" | 17个关键点位置 |
| 可视化 | 文字标签 | 骨骼结构图 |

## 🔗 相关资源
- [OpenPose论文](https://arxiv.org/abs/1812.08008)
- [ControlNet-Aux](https://github.com/patrickvonplaten/controlnet_aux)
- [MediaPipe](https://google.github.io/mediapipe/)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [COCO关键点数据集](https://cocodataset.org/#keypoints-2020)
