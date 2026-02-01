#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态估计摄像头Web服务 - 极速版 🚀
优化策略：降低OpenPose分辨率，跳帧处理，减少传输
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
import threading
import torch

# 保存原始的cv2.imshow
_original_imshow = cv2.imshow

# 使用YOLO进行人体检测
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
    cv2.imshow = _original_imshow
    
    if torch.cuda.is_available():
        yolo_model.to('cuda')
        print("✅ YOLO模型加载成功 (GPU加速)")
    else:
        print("✅ YOLO模型加载成功 (CPU模式)")
    USE_YOLO = True
except Exception as e:
    USE_YOLO = False
    yolo_model = None
    print(f"⚠️ YOLO加载失败: {e}")

# 使用OpenPose进行姿态估计
try:
    from controlnet_aux import OpenposeDetector
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    USE_OPENPOSE = True
    print("✅ OpenPose模型加载成功")
except Exception as e:
    USE_OPENPOSE = False
    pose_detector = None
    print(f"⚠️ OpenPose加载失败: {e}")

BACKGROUND_PATH = r'D:\transformers训练\transformers-main\实战训练\图像任务\姿态估计\背景.png'

print("=" * 70)
print("🚀 姿态估计摄像头Web服务 - 极速版")
print("=" * 70)

# 读取背景图片
background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pose-detection-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
camera = None
camera_lock = threading.Lock()
is_detecting = False
detection_thread = None

# 性能优化参数
DETECT_RESOLUTION = 128  # OpenPose检测分辨率（越小越快）
SKIP_FRAMES = 2  # 跳帧数（2=每2帧检测一次）
JPEG_QUALITY = 60  # JPEG质量（越低越快）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 实时姿态检测 - 极速版</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: linear-gradient(135deg, rgba(255, 140, 0, 0.95) 0%, rgba(255, 100, 50, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 140, 0, 0.5);
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 140, 0, 0.6);
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #ff8c00, #ff4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.1em;
            font-weight: bold;
        }}
        
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        button {{
            padding: 15px 40px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(255, 140, 0, 0.4);
            background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%);
            color: white;
            margin: 0 10px;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 140, 0, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        button.stop-btn {{
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        }}
        
        .video-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .video-box {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            border: 3px solid #fff;
        }}
        
        .video-box h3 {{
            color: #ff8c00;
            margin-bottom: 10px;
            text-align: center;
            font-size: 1.3em;
        }}
        
        .video-frame {{
            width: 100%;
            height: auto;
            border-radius: 15px;
            background: #000;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.2em;
        }}
        
        .video-frame img {{
            width: 100%;
            height: auto;
            border-radius: 15px;
        }}
        
        .stats {{
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.95) 0%, rgba(56, 142, 60, 0.95) 100%);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
            border: 3px solid #fff;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            background: white;
            padding: 15px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            color: #4caf50;
            font-size: 2em;
            font-weight: bold;
        }}
        
        .status {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            color: #666;
            border: 2px solid #ff8c00;
        }}
        
        .status.active {{
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.95) 0%, rgba(56, 142, 60, 0.95) 100%);
            color: white;
            border-color: #4caf50;
        }}
        
        .optimization-info {{
            background: rgba(255, 235, 59, 0.95);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            border: 2px solid #ffc107;
            color: #333;
            text-align: center;
        }}
        
        @media (max-width: 1200px) {{
            .video-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 实时姿态检测 - 极速版</h1>
        <p class="subtitle">GPU加速 + 性能优化 = 流畅体验</p>
        
        <div class="optimization-info">
            ⚡ 优化策略：降低分辨率({DETECT_RESOLUTION}px) + 跳帧处理(1/{SKIP_FRAMES}) + 压缩传输({JPEG_QUALITY}%)
        </div>
        
        <div class="controls">
            <button id="startBtn" onclick="startDetection()">
                📹 启动摄像头
            </button>
            <button id="stopBtn" onclick="stopDetection()" class="stop-btn" style="display: none;">
                ⏹️ 停止检测
            </button>
        </div>
        
        <div id="status" class="status">
            ⏸️ 摄像头未启动
        </div>
        
        <div class="stats" id="stats" style="display: none;">
            <div class="stat-item">
                <div class="stat-label">👥 人数</div>
                <div class="stat-value" id="peopleCount">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">📊 FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">⏱️ 延迟</div>
                <div class="stat-value" id="latency" style="font-size: 1.5em;">0ms</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">🚀 GPU</div>
                <div class="stat-value" id="gpuStatus" style="font-size: 1.2em;">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">🎯 检测率</div>
                <div class="stat-value" id="detectRate" style="font-size: 1.3em;">-</div>
            </div>
        </div>
        
        <div class="video-container">
            <div class="video-box">
                <h3>📷 原始画面</h3>
                <div class="video-frame" id="originalFrame">
                    等待摄像头启动...
                </div>
            </div>
            <div class="video-box">
                <h3>🦴 骨骼检测</h3>
                <div class="video-frame" id="poseFrame">
                    等待检测开始...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let isRunning = false;
        let frameCount = 0;
        let lastTime = Date.now();
        
        socket.on('connect', function() {{
            console.log('✅ WebSocket连接成功');
        }});
        
        socket.on('frame', function(data) {{
            document.getElementById('originalFrame').innerHTML = 
                '<img src="data:image/jpeg;base64,' + data.original + '" alt="原始画面">';
            
            document.getElementById('poseFrame').innerHTML = 
                '<img src="data:image/jpeg;base64,' + data.pose + '" alt="骨骼检测">';
            
            document.getElementById('peopleCount').textContent = data.num_people || 0;
            document.getElementById('latency').textContent = Math.round(data.latency || 0) + 'ms';
            document.getElementById('detectRate').textContent = '1/' + {SKIP_FRAMES};
            
            frameCount++;
            const now = Date.now();
            if (now - lastTime >= 1000) {{
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }}
        }});
        
        socket.on('status', function(data) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = data.message;
            
            if (data.active) {{
                statusDiv.className = 'status active';
            }} else {{
                statusDiv.className = 'status';
            }}
            
            if (data.gpu_available !== undefined) {{
                const gpuStatus = document.getElementById('gpuStatus');
                if (data.gpu_available) {{
                    gpuStatus.textContent = 'ON';
                    gpuStatus.style.color = '#4caf50';
                }} else {{
                    gpuStatus.textContent = 'OFF';
                    gpuStatus.style.color = '#ff9800';
                }}
            }}
        }});
        
        socket.on('error', function(data) {{
            alert('❌ 错误: ' + data.message);
            stopDetection();
        }});
        
        function startDetection() {{
            if (isRunning) return;
            
            socket.emit('start_camera');
            isRunning = true;
            
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('stats').style.display = 'grid';
            
            document.getElementById('status').className = 'status active';
            document.getElementById('status').textContent = '🚀 正在启动摄像头...';
        }}
        
        function stopDetection() {{
            if (!isRunning) return;
            
            socket.emit('stop_camera');
            isRunning = false;
            
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            
            document.getElementById('status').className = 'status';
            document.getElementById('status').textContent = '⏸️ 摄像头已停止';
            
            document.getElementById('originalFrame').textContent = '等待摄像头启动...';
            document.getElementById('poseFrame').textContent = '等待检测开始...';
        }}
        
        window.addEventListener('beforeunload', function() {{
            if (isRunning) {{
                socket.emit('stop_camera');
            }}
        }});
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def detect_pose_in_frame(frame):
    """对单帧图像进行姿态检测（极速优化版）"""
    start_time = time.time()
    
    # 转换为PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # YOLO检测人数（GPU加速）
    num_people = 0
    if USE_YOLO and yolo_model:
        try:
            results = yolo_model(pil_image, verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:
                        num_people += 1
        except Exception as e:
            pass
    
    # OpenPose检测骨骼（极速模式）
    pose_frame = frame.copy()
    if USE_OPENPOSE and pose_detector:
        try:
            # 极速配置：最低分辨率 + 不检测手和脸
            pose_image = pose_detector(
                pil_image, 
                detect_resolution=DETECT_RESOLUTION,  # 128px 极速
                image_resolution=DETECT_RESOLUTION,
                hand_and_face=False,  # 不检测手和脸
                output_type='pil'
            )
            
            pose_array = np.array(pose_image)
            pose_frame = cv2.cvtColor(pose_array, cv2.COLOR_RGB2BGR)
            pose_frame = cv2.resize(pose_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"姿态检测失败: {e}")
    
    # 添加信息
    cv2.putText(pose_frame, f'People: {num_people}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    latency = (time.time() - start_time) * 1000
    
    return pose_frame, num_people, latency

def camera_loop():
    """摄像头循环（极速优化版）"""
    global camera, is_detecting
    
    with camera_lock:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            socketio.emit('error', {'message': '无法打开摄像头'})
            is_detecting = False
            return
    
    # 摄像头配置
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    socketio.emit('status', {
        'message': '🚀 摄像头运行中（极速模式）',
        'active': True,
        'gpu_available': torch.cuda.is_available()
    })
    
    print("✅ 摄像头已启动（极速模式）")
    print(f"   - 检测分辨率: {DETECT_RESOLUTION}px")
    print(f"   - 跳帧率: 1/{SKIP_FRAMES}")
    print(f"   - JPEG质量: {JPEG_QUALITY}%")
    
    frame_count = 0
    last_pose_frame = None
    last_num_people = 0
    last_latency = 0
    
    while is_detecting:
        ret, frame = camera.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break
        
        frame_count += 1
        
        try:
            # 跳帧处理
            if frame_count % SKIP_FRAMES == 0:
                pose_frame, num_people, latency = detect_pose_in_frame(frame)
                last_pose_frame = pose_frame
                last_num_people = num_people
                last_latency = latency
            else:
                # 使用上一帧的检测结果
                if last_pose_frame is not None:
                    pose_frame = last_pose_frame
                    num_people = last_num_people
                    latency = last_latency
                else:
                    pose_frame = frame.copy()
                    num_people = 0
                    latency = 0
            
            # JPEG编码（低质量高速度）
            _, original_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            _, pose_buffer = cv2.imencode('.jpg', pose_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            # Base64编码
            original_base64 = base64.b64encode(original_buffer).decode('utf-8')
            pose_base64 = base64.b64encode(pose_buffer).decode('utf-8')
            
            # 发送到前端
            socketio.emit('frame', {
                'original': original_base64,
                'pose': pose_base64,
                'num_people': num_people,
                'latency': latency
            })
            
            # 最小延迟
            time.sleep(0.001)
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            continue
    
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    
    print("✅ 摄像头已停止")

@socketio.on('start_camera')
def handle_start_camera():
    global is_detecting, detection_thread
    
    if is_detecting:
        emit('status', {'message': '⚠️ 摄像头已在运行中', 'active': True})
        return
    
    is_detecting = True
    detection_thread = threading.Thread(target=camera_loop, daemon=True)
    detection_thread.start()
    
    emit('status', {'message': '🚀 正在启动摄像头...', 'active': True})

@socketio.on('stop_camera')
def handle_stop_camera():
    global is_detecting
    
    is_detecting = False
    emit('status', {'message': '⏸️ 摄像头已停止', 'active': False})

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🚀 启动实时姿态检测服务（极速版）")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6007")
    print("⚡ 极速优化：更低分辨率 + 跳帧处理 + 压缩传输")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 使用CPU模式")
    
    print()
    
    # 延迟1.5秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://localhost:6007')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    socketio.run(app, host='0.0.0.0', port=6007, debug=False, allow_unsafe_werkzeug=True)
