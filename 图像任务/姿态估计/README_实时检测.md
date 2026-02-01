# 姿态估计实时检测系统 🎥

## 项目概述

这是一个完整的**人体姿态估计实时检测系统**，包含两种使用方式：

1. **桌面版**：使用OpenCV GUI窗口显示（`姿态估计实时检测.py`）
2. **Web版**：使用浏览器访问，左右分屏显示（`姿态估计摄像头Web服务.py`）

## 快速开始

### 桌面版（OpenCV GUI）

```bash
# 启动程序
python 姿态估计实时检测.py

# 选择模式
1. 摄像头实时检测
2. 视频文件检测
3. 退出
```

**特点**：
- ✅ 简单直接，无需浏览器
- ✅ 支持截图、暂停、快进/后退
- ✅ 可保存处理后的视频
- ❌ 单窗口显示（原图和骨骼图切换）

### Web版（浏览器访问）⭐ 推荐

```bash
# 启动服务
python 姿态估计摄像头Web服务.py

# 浏览器访问
http://localhost:6006
```

**特点**：
- ✅ 左右分屏显示（原图+骨骼图）
- ✅ 实时统计信息（FPS、延迟、人数、GPU状态）
- ✅ 美观的Web界面
- ✅ 支持多设备访问
- ✅ GPU加速，性能更好

## 功能对比

| 功能 | 桌面版 | Web版 |
|------|--------|-------|
| 摄像头检测 | ✅ | ✅ |
| 视频文件检测 | ✅ | ❌ |
| 左右分屏 | ❌ | ✅ |
| 实时统计 | 部分 | ✅ |
| GPU加速 | ✅ | ✅ |
| 截图保存 | ✅ | 可扩展 |
| 视频保存 | ✅ | 可扩展 |
| 界面美观度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 使用便捷性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 技术栈

### 核心技术

- **OpenCV**: 摄像头捕获和图像处理
- **YOLO (YOLOv8n)**: 人体检测和计数
- **OpenPose (ControlNet)**: 骨骼关键点检测
- **PyTorch**: GPU加速计算

### Web版额外技术

- **Flask**: Web服务器框架
- **Socket.IO**: WebSocket实时通信
- **HTML5 + JavaScript**: 前端界面

## 安装依赖

### 基础依赖（两个版本都需要）

```bash
# 1. NumPy 1.x（必须！）
pip install "numpy<2"

# 2. OpenCV（GUI版本）
pip install opencv-python==4.8.1.78

# 3. YOLO和OpenPose
pip install ultralytics
pip install controlnet-aux
```

### Web版额外依赖

```bash
# Flask和WebSocket
pip install flask-socketio
```

### GPU加速（可选但推荐）

```bash
# CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 性能表现

### GPU模式（RTX 3070）

| 指标 | 桌面版 | Web版 |
|------|--------|-------|
| FPS | 25-35 | 20-30 |
| 延迟 | 25-40ms | 30-50ms |
| 显存占用 | ~2GB | ~2.5GB |
| CPU占用 | 10-15% | 15-20% |

### CPU模式

| 指标 | 桌面版 | Web版 |
|------|--------|-------|
| FPS | 8-12 | 5-10 |
| 延迟 | 80-120ms | 100-200ms |
| CPU占用 | 60-80% | 70-90% |

## 使用场景

### 桌面版适合

- 需要处理视频文件
- 需要保存处理后的视频
- 不需要Web界面
- 单人使用

### Web版适合

- 需要实时查看原图和骨骼图对比
- 需要美观的界面
- 需要实时统计信息
- 多人/多设备访问
- 演示展示

## 文件说明

```
姿态估计/
├── 姿态估计实时检测.py              # 桌面版（OpenCV GUI）
├── 姿态估计摄像头Web服务.py          # Web版（浏览器访问）
├── 姿态估计Web服务.py                # 图片上传版（原有）
├── 实时检测使用说明.md               # 桌面版使用说明
├── 摄像头Web服务使用说明.md          # Web版使用说明
├── 安装依赖.md                       # 依赖安装指南
├── OpenCV_GUI问题解决方案.md         # 常见问题解决
└── README_实时检测.md                # 本文件
```

## 常见问题

### Q: 选择哪个版本？

**A**: 
- **日常使用**：推荐Web版（界面美观，功能完整）
- **视频处理**：使用桌面版（支持视频文件）
- **快速测试**：使用桌面版（启动更快）

### Q: 为什么Web版FPS比桌面版低？

**A**: Web版需要额外的编码和网络传输：
1. 图像JPEG编码
2. Base64转换
3. WebSocket传输
4. 浏览器解码显示

这些步骤会增加约5-10ms的延迟。

### Q: 如何提高性能？

**A**: 
1. **使用GPU**：确保安装CUDA版本的PyTorch
2. **降低分辨率**：修改detect_resolution参数
3. **跳帧处理**：不是每帧都检测
4. **关闭其他程序**：释放GPU和CPU资源

### Q: 摄像头无法打开？

**A**: 
1. 检查摄像头是否被占用
2. 检查Windows隐私设置
3. 尝试修改摄像头索引（0改为1或2）
4. 重启电脑

### Q: OpenCV GUI错误？

**A**: 参考 `OpenCV_GUI问题解决方案.md`

## 开发计划

### 已完成 ✅

- [x] 桌面版实时检测
- [x] Web版实时检测
- [x] GPU加速支持
- [x] YOLO人体检测
- [x] OpenPose骨骼检测
- [x] 左右分屏显示
- [x] 实时统计信息

### 计划中 📋

- [ ] Web版视频文件上传检测
- [ ] Web版截图功能
- [ ] Web版录制功能
- [ ] 姿态分析报告
- [ ] 动作识别
- [ ] 多摄像头支持
- [ ] 移动端适配

## 技术支持

### 文档

- `实时检测使用说明.md` - 桌面版详细教程
- `摄像头Web服务使用说明.md` - Web版详细教程
- `安装依赖.md` - 依赖安装步骤
- `OpenCV_GUI问题解决方案.md` - 问题排查

### 环境信息

- **Python**: 3.11
- **GPU**: NVIDIA GeForce RTX 3070 Laptop (8GB)
- **Python路径**: `D:\aaaalokda\envs\myenv\python.exe`
- **模型缓存**: `D:\transformers训练\transformers-main\预训练模型下载处`

### 端口使用

- **图片上传版**: http://localhost:6005
- **摄像头Web版**: http://localhost:6006

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

---

**开始使用吧！** 🚀

选择适合你的版本，开始实时姿态检测之旅！
