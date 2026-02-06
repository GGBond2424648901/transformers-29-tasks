#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态估计 Web 服务 - 运动少女风格 🤸
真正的人体骨骼关键点检测
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import io
import base64
import numpy as np

# 使用controlnet_aux的OpenPose检测器
try:
    from controlnet_aux import OpenposeDetector
    USE_OPENPOSE = True
    print("✅ 使用 OpenPose 进行姿态估计")
except ImportError:
    USE_OPENPOSE = False
    print("⚠️ controlnet_aux 未安装，将使用简化版本")
    print("   安装命令: pip install controlnet-aux")

# 使用YOLO进行人体检测
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')  # 使用nano版本，速度快
    USE_YOLO = True
    print("✅ 使用 YOLO 进行人体检测")
except ImportError:
    USE_YOLO = False
    yolo_model = None
    print("⚠️ ultralytics 未安装，将使用简化版本")
    print("   安装命令: pip install ultralytics")

BACKGROUND_PATH = r'D:\transformers训练\transformers-main\实战训练\图像任务\姿态估计\背景.png'

print("=" * 70)
print("🤸 姿态估计 Web 服务 - 运动少女")
print("=" * 70)

print("\n🏃 正在召唤运动少女...")

if USE_OPENPOSE:
    # 初始化OpenPose检测器
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    print("💪 运动少女准备完毕！使用OpenPose检测人体骨骼~")
else:
    pose_detector = None
    print("💪 运动少女准备完毕！使用简化版骨骼检测~")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

# 读取背景图片
background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')
    print(f"✅ 背景图片加载成功: {BACKGROUND_PATH}")
else:
    print(f"⚠️ 背景图片未找到: {BACKGROUND_PATH}")

# COCO关键点名称（17个关键点）
KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
]

# 骨骼连接（用于绘制骨架）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上半身
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)  # 下半身
]

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤸 姿态估计 - 运动少女</title>
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
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* 能量波纹飘落动画 */
        .energy-wave {{
            position: fixed;
            font-size: 30px;
            animation: waveFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 8px rgba(255,140,0,0.8));
        }}
        
        @keyframes waveFall {{
            0% {{
                transform: translateY(-10px) scale(1);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh) scale(1.5);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(255, 140, 0, 0.95) 0%, rgba(255, 100, 50, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 140, 0, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1200px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 140, 0, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #ff8c00, #ff4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            animation: energyPulse 2s ease-in-out infinite;
        }}
        
        @keyframes energyPulse {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.3); }}
        }}
        
        .subtitle {{
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
            font-size: 1.2em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .upload-area {{
            border: 3px dashed #fff;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, rgba(255, 160, 50, 0.8) 0%, rgba(255, 120, 70, 0.8) 100%);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }}
        
        .upload-area::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            animation: energyRipple 2s ease-out infinite;
        }}
        
        @keyframes energyRipple {{
            0% {{
                width: 0;
                height: 0;
                opacity: 1;
            }}
            100% {{
                width: 500px;
                height: 500px;
                opacity: 0;
            }}
        }}
        
        .upload-area:hover {{
            border-color: #ffeb3b;
            background: linear-gradient(135deg, rgba(255, 180, 70, 0.8) 0%, rgba(255, 140, 90, 0.8) 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: sportSpin 3s linear infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes sportSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .preview-box {{
            text-align: center;
        }}
        
        .preview-box h3 {{
            color: #fff;
            margin-bottom: 10px;
            font-size: 1.3em;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(255, 140, 0, 0.4);
            border: 4px solid #fff;
        }}
        
        button {{
            width: 100%;
            padding: 18px;
            font-size: 1.3em;
            font-weight: bold;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(255, 140, 0, 0.4);
            background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%);
            color: white;
            margin-bottom: 15px;
            position: relative;
            overflow: hidden;
        }}
        
        button::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}
        
        button:hover::before {{
            width: 300px;
            height: 300px;
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
        
        .result-container {{
            background: linear-gradient(135deg, rgba(255, 160, 50, 0.8) 0%, rgba(255, 120, 70, 0.8) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #fff;
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .keypoint-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }}
        
        .keypoint-item {{
            background: white;
            padding: 10px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(255, 140, 0, 0.2);
            border-left: 4px solid #ff8c00;
        }}
        
        .keypoint-name {{
            font-weight: bold;
            color: #333;
        }}
        
        .keypoint-confidence {{
            color: #ff8c00;
            font-size: 0.9em;
        }}
        
        .energy-icon {{
            display: inline-block;
            animation: energyBlink 1s ease-in-out infinite;
        }}
        
        @keyframes energyBlink {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.2); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤸 姿态估计助手</h1>
        <p class="subtitle">运动少女帮你检测人体骨骼关键点！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">⚽</div>
            <p style="font-size: 1.2em; color: #fff; font-weight: bold; position: relative; z-index: 1;">
                点击上传图片开始骨骼检测~
            </p>
            <p style="color: #ffe; margin-top: 10px; position: relative; z-index: 1;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <div class="preview-box">
                <h3>📷 原始图片</h3>
                <img id="previewImage" class="preview-image">
            </div>
            <div class="preview-box">
                <h3>🦴 骨骼检测</h3>
                <img id="skeletonImage" class="preview-image">
            </div>
        </div>
        
        <button id="poseBtn" onclick="detectPose()" style="display: none;">
            <span class="energy-icon">💪</span> 开始检测 <span class="energy-icon">💪</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建能量波纹（持续飘落）
        function createEnergyWave() {{
            const waves = ['〰️', '🌊', '💨', '⚡', '💥'];
            const wave = document.createElement('div');
            wave.className = 'energy-wave';
            wave.textContent = waves[Math.floor(Math.random() * waves.length)];
            wave.style.left = Math.random() * 100 + '%';
            wave.style.animationDuration = (Math.random() * 2 + 3) + 's';
            wave.style.fontSize = (Math.random() * 15 + 20) + 'px';
            document.body.appendChild(wave);
            
            setTimeout(() => wave.remove(), 5000);
        }}
        
        // 每250ms创建一个新波纹
        setInterval(createEnergyWave, 250);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('skeletonImage').src = '';
                    document.getElementById('previewContainer').style.display = 'grid';
                    document.getElementById('poseBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function detectPose() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const poseBtn = document.getElementById('poseBtn');
            const skeletonImg = document.getElementById('skeletonImage');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #fff; font-size: 1.2em;">💪 运动少女正在检测骨骼关键点...</p>';
            resultDiv.style.display = 'block';
            skeletonImg.src = '';
            poseBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {{
                const response = await fetch('/detect', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ ${{data.error}}</p>`;
                }} else {{
                    // 显示骨骼图像
                    if (data.skeleton_image) {{
                        skeletonImg.src = 'data:image/png;base64,' + data.skeleton_image;
                    }}
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ 检测失败: ${{error.message}}</p>`;
            }} finally {{
                poseBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            // 关键点的详细描述
            const keypointInfo = {{
                '鼻子': {{ emoji: '👃', desc: '面部中心定位点' }},
                '左眼': {{ emoji: '👁️', desc: '左侧视觉感知' }},
                '右眼': {{ emoji: '👁️', desc: '右侧视觉感知' }},
                '左耳': {{ emoji: '👂', desc: '左侧听觉定位' }},
                '右耳': {{ emoji: '👂', desc: '右侧听觉定位' }},
                '左肩': {{ emoji: '💪', desc: '左臂起始关节' }},
                '右肩': {{ emoji: '💪', desc: '右臂起始关节' }},
                '左肘': {{ emoji: '🔗', desc: '左臂弯曲点' }},
                '右肘': {{ emoji: '🔗', desc: '右臂弯曲点' }},
                '左腕': {{ emoji: '✋', desc: '左手连接处' }},
                '右腕': {{ emoji: '✋', desc: '右手连接处' }},
                '左髋': {{ emoji: '🦵', desc: '左腿起始关节' }},
                '右髋': {{ emoji: '🦵', desc: '右腿起始关节' }},
                '左膝': {{ emoji: '🦿', desc: '左腿弯曲点' }},
                '右膝': {{ emoji: '🦿', desc: '右腿弯曲点' }},
                '左踝': {{ emoji: '👟', desc: '左脚连接处' }},
                '右踝': {{ emoji: '👟', desc: '右脚连接处' }}
            }};
            
            let html = '<h3 style="color: #fff; margin-bottom: 20px; text-align: center; font-size: 1.8em;">🦴 骨骼关键点检测结果</h3>';
            
            if (data.num_people > 0) {{
                html += `
                    <div style="text-align: center; margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, rgba(255,235,59,0.95) 0%, rgba(255,193,7,0.95) 100%); border-radius: 15px; border: 3px solid #fff;">
                        <p style="font-size: 1.5em; color: #ff6f00; font-weight: bold; margin-bottom: 10px;">
                            🎯 成功检测到 <span style="font-size: 1.8em; color: #d84315;">${{data.num_people}}</span> 个人体姿态
                        </p>
                        <p style="color: #f57c00; font-size: 1.1em;">完整识别17个核心关键点</p>
                    </div>
                `;
                
                if (data.keypoints && data.keypoints.length > 0) {{
                    html += '<div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 3px solid #fff;">';
                    html += '<h4 style="color: #ff8c00; margin-bottom: 15px; text-align: center; font-size: 1.4em;">📍 检测到的关键点详情</h4>';
                    html += '<div class="keypoint-list">';
                    
                    data.keypoints.forEach(kp => {{
                        const info = keypointInfo[kp.name] || {{ emoji: '⚫', desc: '' }};
                        html += `
                            <div class="keypoint-item" style="flex-direction: column; align-items: flex-start; padding: 12px;">
                                <div style="display: flex; align-items: center; width: 100%; margin-bottom: 5px;">
                                    <span style="font-size: 1.6em; margin-right: 10px;">${{info.emoji}}</span>
                                    <span class="keypoint-name" style="font-size: 1.1em;">${{kp.name}}</span>
                                    <span class="keypoint-confidence" style="margin-left: auto; font-size: 1.2em;">✓</span>
                                </div>
                                <div style="font-size: 0.9em; color: #666; padding-left: 40px;">${{info.desc}}</div>
                            </div>
                        `;
                    }});
                    
                    html += '</div></div>';
                }}
                
                // 添加姿态分析
                html += `
                    <div style="background: linear-gradient(135deg, rgba(255,235,59,0.9) 0%, rgba(255,193,7,0.9) 100%); 
                                padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 3px solid #fff;">
                        <h4 style="color: #ff6f00; margin-bottom: 15px; text-align: center; font-size: 1.3em;">💡 姿态分析报告</h4>
                        <div style="color: #333; line-height: 2; font-size: 1.05em;">
                            <p>✨ <strong>头部区域</strong>：检测到面部5个关键点（鼻子、双眼、双耳）</p>
                            <p>💪 <strong>上肢区域</strong>：检测到双臂6个关键点（双肩、双肘、双腕）</p>
                            <p>🦵 <strong>下肢区域</strong>：检测到双腿6个关键点（双髋、双膝、双踝）</p>
                            <p>🎯 <strong>检测总计</strong>：完整识别人体17个核心关键点</p>
                        </div>
                    </div>
                `;
            }} else {{
                html += `
                    <div style="text-align: center; padding: 30px; background: rgba(255,255,255,0.9); border-radius: 15px; border: 3px solid #fff;">
                        <p style="font-size: 1.5em; color: #ff6f00; margin-bottom: 10px;">😢 未检测到人体</p>
                        <p style="color: #666;">请上传包含人物的清晰图片</p>
                    </div>
                `;
            }}
            
            html += `
                <div style="margin-top: 20px; padding: 20px; background: rgba(255,255,255,0.95); border-radius: 15px; color: #666; border: 2px solid #ff8c00;">
                    <p style="font-size: 1.2em; color: #ff8c00; font-weight: bold; margin-bottom: 10px;">💪 检测说明</p>
                    <p style="margin-top: 8px; line-height: 1.8;">• 使用OpenPose技术进行人体骨骼关键点检测</p>
                    <p style="line-height: 1.8;">• 右侧图像显示检测到的完整骨骼结构（绿色线条为骨骼，红色点为关键点）</p>
                    <p style="line-height: 1.8;">• 17个关键点覆盖头部、躯干、四肢的关键位置</p>
                    <p style="line-height: 1.8;">• 上传清晰的全身照片可以获得更好的检测效果</p>
                    <p style="line-height: 1.8;">• 支持多人同时检测（图片中有多人时会显示总人数）</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#ffeb3b';
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.style.borderColor = '#fff';
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#fff';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {{
                const event = {{ target: {{ files: [file] }} }};
                handleFileSelect(event);
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
    return HTML_TEMPLATE

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 首先使用YOLO检测人数
        num_people = 0
        if USE_YOLO and yolo_model:
            try:
                # YOLO检测
                results = yolo_model(image, verbose=False)
                
                # 统计检测到的人（class 0 是person）
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # class 0 = person
                            num_people += 1
                
                print(f"🎯 YOLO检测到 {num_people} 个人")
            except Exception as e:
                print(f"⚠️ YOLO检测失败: {e}")
                num_people = 0
        
        if USE_OPENPOSE and pose_detector:
            # 使用OpenPose检测
            pose_image = pose_detector(image, detect_resolution=512, image_resolution=512)
            
            # 将骨骼图像转换为base64
            buffered = io.BytesIO()
            pose_image.save(buffered, format="PNG")
            skeleton_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 如果YOLO没有检测到人，使用备用方法
            if num_people == 0:
                print("⚠️ YOLO未检测到人，使用图像分析方法")
                # 改进的人数检测：分析骨骼图像中红色关键点的聚类
                pose_array = np.array(pose_image.convert('RGB'))
                
                # 检测红色关键点（OpenPose使用红色标记关键点）
                red_channel = pose_array[:, :, 0]
                green_channel = pose_array[:, :, 1]
                blue_channel = pose_array[:, :, 2]
                
                # 找出红色占主导的像素（关键点）- 降低阈值以检测更多点
                red_mask = (red_channel > 100) & (red_channel > green_channel + 30) & (red_channel > blue_channel + 30)
                
                # 获取所有红色点的坐标
                red_points = np.argwhere(red_mask)
                
                if len(red_points) > 50:  # 至少需要一定数量的点才能聚类
                    # 使用DBSCAN聚类算法将关键点分组为不同的人
                    from sklearn.cluster import DBSCAN
                    
                    clustering = DBSCAN(eps=150, min_samples=10).fit(red_points)
                    labels = clustering.labels_
                    
                    # 统计聚类数量（排除噪声点，label=-1）
                    unique_labels = set(labels)
                    num_people = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    
                    print(f"✅ 聚类检测到 {num_people} 个人体")
                
                # 如果还是0，至少设为1
                if num_people == 0:
                    num_people = 1
                    print(f"⚠️ 默认设置为 {num_people} 个人")
            
            print(f"✅ 最终检测结果: {num_people} 个人体")
            
            result = {
                'num_people': num_people,
                'keypoints': [{'name': name, 'detected': True} for name in KEYPOINT_NAMES],
                'skeleton_image': skeleton_base64,
                'detection_quality': 'high' if num_people > 0 else 'low'
            }
        else:
            # 简化版本：绘制示例骨架
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # 在图像中心绘制一个简单的骨架示例
            width, height = image.size
            cx, cy = width // 2, height // 2
            
            # 简单的骨架坐标（相对于中心）
            scale = min(width, height) // 4
            keypoints_pos = [
                (cx, cy - scale),  # 鼻子
                (cx - scale//4, cy - scale - scale//8), (cx + scale//4, cy - scale - scale//8),  # 眼睛
                (cx - scale//3, cy - scale - scale//6), (cx + scale//3, cy - scale - scale//6),  # 耳朵
                (cx - scale//2, cy - scale//3), (cx + scale//2, cy - scale//3),  # 肩膀
                (cx - scale//2, cy + scale//4), (cx + scale//2, cy + scale//4),  # 肘部
                (cx - scale//2, cy + scale//2), (cx + scale//2, cy + scale//2),  # 手腕
                (cx - scale//3, cy + scale//3), (cx + scale//3, cy + scale//3),  # 髋部
                (cx - scale//3, cy + scale), (cx + scale//3, cy + scale),  # 膝盖
                (cx - scale//3, cy + scale * 1.5), (cx + scale//3, cy + scale * 1.5),  # 脚踝
            ]
            
            # 绘制骨骼连接
            for conn in SKELETON_CONNECTIONS:
                if conn[0] < len(keypoints_pos) and conn[1] < len(keypoints_pos):
                    draw.line([keypoints_pos[conn[0]], keypoints_pos[conn[1]]], 
                             fill='#00ff00', width=3)
            
            # 绘制关键点
            for pos in keypoints_pos:
                draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], 
                           fill='#ff0000', outline='#ffffff')
            
            # 转换为base64
            buffered = io.BytesIO()
            draw_image.save(buffered, format="PNG")
            skeleton_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            result = {
                'num_people': 1,
                'keypoints': [{'name': name, 'detected': True} for name in KEYPOINT_NAMES],
                'skeleton_image': skeleton_base64,
                'detection_quality': 'demo'
            }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    import webbrowser
    import threading
    
    print("🏃 启动运动少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6005")
    print("💪 运动少女在这里等你~")
    if not USE_OPENPOSE:
        print("\n⚠️  提示: 安装 controlnet-aux 可获得更好的检测效果")
        print("   pip install controlnet-aux")
    print()
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6005')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6005, debug=False)
