#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态估计摄像头Web服务 - 实时检测 🎥
支持GPU加速，左右分屏显示
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, render_template_string, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
import threading
import torch

# 保存原始的cv2.imshow，避免被ultralytics patch
_original_imshow = cv2.imshow

# 使用YOLO进行人体检测
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
    # 恢复原始的imshow函数
    cv2.imshow = _original_imshow
    
    # 如果有GPU，使用GPU
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
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载OpenPose并指定设备
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    # 将模型移到GPU（如果可用）
    if hasattr(pose_detector, 'model_pose') and device == 'cuda':
        pose_detector.model_pose = pose_detector.model_pose.to(device)
        print(f"✅ OpenPose模型加载成功 (GPU加速)")
    else:
        print(f"✅ OpenPose模型加载成功 (CPU模式)")
    
    USE_OPENPOSE = True
except Exception as e:
    USE_OPENPOSE = False
    pose_detector = None
    print(f"⚠️ OpenPose加载失败: {e}")

BACKGROUND_PATH = r'D:\transformers训练\transformers-main\实战训练\图像任务\姿态估计\背景.png'

print("=" * 70)
print("🎥 姿态估计摄像头Web服务 - 实时检测")
print("=" * 70)

# 读取背景图片
background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')
    print(f"✅ 背景图片加载成功")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pose-detection-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
camera = None
camera_lock = threading.Lock()
is_detecting = False
detection_thread = None

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎥 实时姿态检测 - 摄像头</title>
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
            overflow-x: hidden;
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
            animation: energyPulse 2s ease-in-out infinite;
        }}
        
        @keyframes energyPulse {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.3); }}
        }}
        
        .subtitle {{
            text-align: center;
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.1em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
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
            position: relative;
            overflow: hidden;
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
            background: linear-gradient(135deg, rgba(255, 235, 59, 0.95) 0%, rgba(255, 193, 7, 0.95) 100%);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
            border: 3px solid #fff;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            color: #ff8c00;
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
        
        .loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
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
        <h1>🎥 实时姿态检测</h1>
        <p class="subtitle">摄像头实时骨骼关键点检测 - GPU加速</p>
        
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
                <div class="stat-label">检测人数</div>
                <div class="stat-value" id="peopleCount">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">处理帧率 (FPS)</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">检测延迟 (ms)</div>
                <div class="stat-value" id="latency">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">GPU状态</div>
                <div class="stat-value" id="gpuStatus" style="font-size: 1.2em;">-</div>
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
            // 更新原始画面
            document.getElementById('originalFrame').innerHTML = 
                '<img src="data:image/jpeg;base64,' + data.original + '" alt="原始画面">';
            
            // 更新骨骼检测画面
            document.getElementById('poseFrame').innerHTML = 
                '<img src="data:image/jpeg;base64,' + data.pose + '" alt="骨骼检测">';
            
            // 更新统计信息
            document.getElementById('peopleCount').textContent = data.num_people || 0;
            document.getElementById('latency').textContent = Math.round(data.latency || 0);
            
            // 计算FPS
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
            
            // 更新GPU状态
            if (data.gpu_available !== undefined) {{
                const gpuStatus = document.getElementById('gpuStatus');
                if (data.gpu_available) {{
                    gpuStatus.textContent = '🚀 GPU';
                    gpuStatus.style.color = '#4caf50';
                }} else {{
                    gpuStatus.textContent = '💻 CPU';
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
            document.getElementById('status').innerHTML = 
                '<span class="loading"></span> 正在启动摄像头...';
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
        
        // 页面关闭时停止检测
        window.addEventListener('beforeunload', function() {{
            if (isRunning) {{
                socket.emit('stop_camera');
            }}
        }});
    </script>
</body>
</html>
"""


@app.route('/static/background')
def background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    else:
        return '', 404

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def detect_pose_in_frame(frame):
    """对单帧图像进行姿态检测（优化版）"""
    start_time = time.time()
    
    # 转换为PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # YOLO检测人数（GPU加速）
    num_people = 0
    if USE_YOLO and yolo_model:
        try:
            # YOLO已经在GPU上，直接推理
            results = yolo_model(pil_image, verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # person
                        num_people += 1
        except Exception as e:
            print(f"YOLO检测失败: {e}")
    
    # OpenPose检测骨骼（GPU加速）
    pose_frame = frame.copy()
    if USE_OPENPOSE and pose_detector:
        try:
            # 使用更低的分辨率以提高速度（从256降到192）
            # hand_and_face=False 可以进一步提速
            pose_image = pose_detector(
                pil_image, 
                detect_resolution=192,  # 降低分辨率提速
                image_resolution=192,
                hand_and_face=False,  # 不检测手和脸，提速
                output_type='pil'
            )
            
            # 转换回OpenCV格式
            pose_array = np.array(pose_image)
            pose_frame = cv2.cvtColor(pose_array, cv2.COLOR_RGB2BGR)
            
            # 调整回原始尺寸
            pose_frame = cv2.resize(pose_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"姿态检测失败: {e}")
    
    # 在骨骼图上添加信息
    cv2.putText(pose_frame, f'People: {num_people}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    latency = (time.time() - start_time) * 1000  # 转换为毫秒
    
    return pose_frame, num_people, latency

def camera_loop():
    """摄像头循环（优化版）"""
    global camera, is_detecting
    
    with camera_lock:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            socketio.emit('error', {'message': '无法打开摄像头'})
            is_detecting = False
            return
    
    # 设置摄像头参数（降低分辨率以提高速度）
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延迟
    
    socketio.emit('status', {
        'message': '🎥 摄像头运行中...',
        'active': True,
        'gpu_available': torch.cuda.is_available()
    })
    
    print("✅ 摄像头已启动")
    
    frame_count = 0
    skip_frames = 1  # 每N帧处理一次（1=不跳帧，2=每2帧处理一次）
    
    while is_detecting:
        ret, frame = camera.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break
        
        frame_count += 1
        
        try:
            # 跳帧处理（可选，进一步提速）
            if frame_count % skip_frames == 0:
                # 检测姿态
                pose_frame, num_people, latency = detect_pose_in_frame(frame)
            else:
                # 跳过的帧直接使用原图
                pose_frame = frame.copy()
                num_people = 0
                latency = 0
            
            # 转换为JPEG（降低质量以减少传输时间）
            _, original_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _, pose_buffer = cv2.imencode('.jpg', pose_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            # 转换为base64
            original_base64 = base64.b64encode(original_buffer).decode('utf-8')
            pose_base64 = base64.b64encode(pose_buffer).decode('utf-8')
            
            # 发送到前端
            socketio.emit('frame', {
                'original': original_base64,
                'pose': pose_base64,
                'num_people': num_people,
                'latency': latency
            })
            
            # 控制帧率（减少sleep时间以提高响应速度）
            time.sleep(0.01)  # 约100fps（实际受检测速度限制）
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 释放摄像头
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    
    print("✅ 摄像头已停止")

@socketio.on('start_camera')
def handle_start_camera():
    """启动摄像头"""
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
    """停止摄像头"""
    global is_detecting
    
    is_detecting = False
    emit('status', {'message': '⏸️ 摄像头已停止', 'active': False})

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎥 启动实时姿态检测服务...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6006")
    print("💪 支持GPU加速，实时检测人体骨骼~")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 使用CPU模式（建议使用GPU以获得更好性能）")
    
    if not USE_YOLO or not USE_OPENPOSE:
        print("\n⚠️  提示: 确保已安装所有依赖")
        print("   pip install ultralytics controlnet-aux flask-socketio")
    
    print()
    
    # 延迟1.5秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://localhost:6006')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    socketio.run(app, host='0.0.0.0', port=6006, debug=False, allow_unsafe_werkzeug=True)
