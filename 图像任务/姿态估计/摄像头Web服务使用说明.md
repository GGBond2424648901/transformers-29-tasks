# 🎥 姿态估计摄像头Web服务使用说明

## 功能介绍

这是一个**实时摄像头姿态检测Web服务**，支持：

- 📹 **实时摄像头检测**：打开电脑摄像头，实时显示骨骼关键点
- 🖥️ **左右分屏显示**：左边显示原始画面，右边显示骨骼检测结果
- 🚀 **GPU加速**：自动使用GPU加速处理，大幅提升帧率
- 📊 **实时统计**：显示检测人数、FPS、延迟、GPU状态
- 🌐 **Web界面**：通过浏览器访问，无需安装客户端

## 技术特点

### 性能优化

1. **GPU加速**：YOLO和OpenPose都支持GPU加速
2. **分辨率优化**：OpenPose使用256x256分辨率，平衡速度和精度
3. **WebSocket实时传输**：使用Socket.IO实现低延迟视频流
4. **多线程处理**：摄像头读取和检测处理分离

### 显示效果

- **左侧画面**：原始摄像头画面（640x480）
- **右侧画面**：骨骼检测结果（绿色线条+红色关键点）
- **实时统计**：
  - 检测人数（YOLO精准计数）
  - 处理帧率（FPS）
  - 检测延迟（毫秒）
  - GPU状态（GPU/CPU）

## 安装依赖

**重要：必须按顺序安装！**

```bash
# 1. NumPy 1.x（必须）
pip install "numpy<2"

# 2. OpenCV（GUI版本）
pip install opencv-python==4.8.1.78

# 3. YOLO和OpenPose
pip install ultralytics
pip install controlnet-aux

# 4. Flask和WebSocket
pip install flask-socketio
```

## 启动服务

### 方法1: 直接运行

```bash
python 姿态估计摄像头Web服务.py
```

### 方法2: 使用完整路径

```bash
D:\aaaalokda\envs\myenv\python.exe "姿态估计摄像头Web服务.py"
```

### 启动信息

```
======================================================================
🎥 姿态估计摄像头Web服务 - 实时检测
======================================================================

✅ YOLO模型加载成功 (GPU加速)
✅ OpenPose模型加载成功
✅ 背景图片加载成功

======================================================================
🎥 启动实时姿态检测服务...
======================================================================

📍 访问地址: http://localhost:6006
💪 支持GPU加速，实时检测人体骨骼~
🚀 GPU: NVIDIA GeForce RTX 3070 Laptop
💾 显存: 8.0 GB
```

## 使用方法

### 1. 打开浏览器

访问：`http://localhost:6006`

### 2. 启动摄像头

点击 **"📹 启动摄像头"** 按钮

- 首次使用需要授权摄像头访问权限
- 等待摄像头启动（约2-3秒）
- 左右两侧会同时显示画面

### 3. 查看实时检测

- **左侧**：原始摄像头画面
- **右侧**：骨骼检测结果
  - 绿色线条：骨骼连接
  - 红色点：关键点位置
  - 显示检测人数

### 4. 查看统计信息

顶部显示实时统计：

| 统计项 | 说明 |
|--------|------|
| 检测人数 | YOLO检测到的人数 |
| 处理帧率 | 每秒处理的帧数（FPS） |
| 检测延迟 | 单帧处理时间（毫秒） |
| GPU状态 | 🚀 GPU 或 💻 CPU |

### 5. 停止检测

点击 **"⏹️ 停止检测"** 按钮

- 摄像头会立即停止
- 释放所有资源
- 可以重新启动

## 性能参考

### GPU模式（RTX 3070）

- **FPS**: 20-30 帧/秒
- **延迟**: 30-50 毫秒/帧
- **人数检测**: 实时准确
- **骨骼检测**: 流畅清晰

### CPU模式

- **FPS**: 5-10 帧/秒
- **延迟**: 100-200 毫秒/帧
- **人数检测**: 实时准确
- **骨骼检测**: 略有延迟

## 优化建议

### 提高帧率

1. **使用GPU**：确保安装CUDA版本的PyTorch
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **降低分辨率**：修改代码中的检测分辨率
   ```python
   # 从256降低到128
   pose_image = pose_detector(pil_image, detect_resolution=128, image_resolution=128)
   ```

3. **调整帧率**：修改sleep时间
   ```python
   # 从0.03改为0.05（降低帧率，减少计算）
   time.sleep(0.05)  # 约20fps
   ```

### 降低延迟

1. **使用更小的YOLO模型**：已使用yolov8n.pt（最小版本）
2. **跳帧检测**：不是每帧都检测
   ```python
   if frame_count % 2 == 0:  # 每2帧检测一次
       pose_frame, num_people, latency = detect_pose_in_frame(frame)
   ```

### 提高准确度

1. **提高分辨率**：
   ```python
   pose_image = pose_detector(pil_image, detect_resolution=512, image_resolution=512)
   ```

2. **改善光线**：确保环境光线充足
3. **调整摄像头**：确保人物完整入镜

## 常见问题

### Q1: 摄像头无法打开？

**症状**：点击启动后显示"无法打开摄像头"

**解决方案**：
1. 检查摄像头是否被其他程序占用
2. 检查Windows隐私设置中的摄像头权限
3. 尝试修改摄像头索引：
   ```python
   camera = cv2.VideoCapture(1)  # 改为1或2
   ```

### Q2: 画面很卡？

**症状**：FPS很低，画面延迟严重

**解决方案**：
1. 确认GPU是否正常工作（查看GPU状态）
2. 降低检测分辨率（见优化建议）
3. 关闭其他占用GPU的程序
4. 检查网络连接（WebSocket传输）

### Q3: 检测不准确？

**症状**：人数统计错误，骨骼位置不对

**解决方案**：
1. 改善光线条件
2. 确保人物完整入镜
3. 避免复杂背景
4. 提高检测分辨率

### Q4: GPU状态显示CPU？

**症状**：明明有GPU，但显示CPU模式

**解决方案**：
1. 检查PyTorch是否支持CUDA：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该返回True
   ```

2. 安装CUDA版本的PyTorch：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. 检查CUDA驱动是否正常

### Q5: 浏览器无法连接？

**症状**：访问localhost:6006无响应

**解决方案**：
1. 检查服务是否正常启动
2. 检查防火墙设置
3. 尝试使用127.0.0.1:6006
4. 检查端口是否被占用：
   ```bash
   netstat -ano | findstr 6006
   ```

### Q6: WebSocket连接失败？

**症状**：浏览器控制台显示WebSocket错误

**解决方案**：
1. 刷新页面重新连接
2. 检查flask-socketio是否正确安装
3. 尝试使用其他浏览器（推荐Chrome）

## 技术细节

### 架构设计

```
浏览器 (HTML5 + WebSocket)
    ↓
Flask + SocketIO (Web服务器)
    ↓
摄像头线程 (OpenCV)
    ↓
检测处理 (YOLO + OpenPose + GPU)
    ↓
Base64编码 (JPEG压缩)
    ↓
WebSocket推送 (实时传输)
```

### 数据流程

1. **摄像头捕获**：OpenCV读取摄像头帧（640x480）
2. **YOLO检测**：检测人体位置和数量（GPU加速）
3. **OpenPose检测**：生成骨骼关键点（256x256）
4. **图像编码**：转换为JPEG并Base64编码
5. **WebSocket推送**：发送到浏览器显示
6. **循环处理**：约30fps的速度持续处理

### 关键技术

- **OpenCV**: 摄像头捕获和图像处理
- **YOLO**: 人体检测和计数
- **OpenPose**: 骨骼关键点检测
- **Flask**: Web服务器框架
- **Socket.IO**: WebSocket实时通信
- **PyTorch**: GPU加速计算

## 扩展功能

### 添加录制功能

可以添加视频录制功能：

```python
# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480))

# 在循环中写入帧
out.write(pose_frame)

# 停止时释放
out.release()
```

### 添加截图功能

在前端添加截图按钮：

```javascript
function captureFrame() {
    const canvas = document.createElement('canvas');
    const img = document.getElementById('poseFrame').querySelector('img');
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
    
    const link = document.createElement('a');
    link.download = 'pose_' + Date.now() + '.png';
    link.href = canvas.toDataURL();
    link.click();
}
```

### 添加姿态分析

基于关键点坐标计算：

```python
def analyze_pose(keypoints):
    # 计算关节角度
    # 检测特定姿态
    # 统计运动数据
    pass
```

## 应用场景

### 健身指导

- 实时监测运动姿态
- 纠正动作错误
- 记录训练数据

### 动作捕捉

- 游戏开发
- 动画制作
- 虚拟现实

### 安全监控

- 人员计数
- 行为分析
- 异常检测

### 体育分析

- 运动员训练
- 技术动作分析
- 比赛数据统计

## 注意事项

1. **隐私保护**：摄像头数据仅在本地处理，不上传到服务器
2. **资源占用**：GPU模式会占用显存，注意其他程序的使用
3. **网络要求**：WebSocket需要稳定的网络连接
4. **浏览器兼容**：推荐使用Chrome、Edge等现代浏览器
5. **摄像头权限**：首次使用需要授权摄像头访问

## 环境信息

- **Python**: 3.11
- **操作系统**: Windows
- **GPU**: NVIDIA GeForce RTX 3070 Laptop (8GB VRAM)
- **Python路径**: `D:\aaaalokda\envs\myenv\python.exe`
- **服务端口**: 6006
- **WebSocket**: Socket.IO 4.5.4

## 更新日志

**v1.0** (2026-02-01)
- ✅ 支持实时摄像头检测
- ✅ 左右分屏显示
- ✅ GPU加速处理
- ✅ WebSocket实时传输
- ✅ 实时统计信息
- ✅ YOLO人体检测
- ✅ OpenPose骨骼检测

---

**祝你使用愉快！** 🎉

如有问题，请查看常见问题部分或检查控制台输出。
