#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频分类 Web 服务 - 偶像少女风格 🎬
支持真实视频文件上传和分类
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2
import tempfile
from googletrans import Translator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🎬 视频分类 Web 服务 - 偶像少女")
print("=" * 70)

print("\n🎤 正在召唤偶像少女...")
# 使用已经fine-tuned的VideoMAE模型（Kinetics-400数据集）
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化翻译器
translator = Translator()
print("🌟 偶像少女准备完毕！开始分类视频~")

app = Flask(__name__)

background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 视频分类 - 偶像少女</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Comic Sans MS', cursive;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* 星光闪耀飘落动画 */
        .star-sparkle {{
            position: fixed;
            font-size: 25px;
            animation: starFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 8px rgba(255,192,203,0.8));
        }}
        
        @keyframes starFall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh) rotate(720deg) scale(0.5);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(255, 182, 193, 0.95) 0%, rgba(221, 160, 221, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 105, 180, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 105, 180, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .idol-icon {{
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 80px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            animation: idolDance 2s ease-in-out infinite;
        }}
        
        @keyframes idolDance {{
            0%, 100% {{ transform: translateX(-50%) translateY(0) rotate(-5deg); }}
            25% {{ transform: translateX(-50%) translateY(-10px) rotate(5deg); }}
            75% {{ transform: translateX(-50%) translateY(-5px) rotate(-5deg); }}
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #ff69b4, #da70d6, #ff1493);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            animation: idolShine 3s ease-in-out infinite;
        }}
        
        @keyframes idolShine {{
            0%, 100% {{ filter: hue-rotate(0deg) brightness(1); }}
            50% {{ filter: hue-rotate(20deg) brightness(1.2); }}
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
            background: linear-gradient(135deg, rgba(255, 192, 203, 0.8) 0%, rgba(221, 160, 221, 0.8) 100%);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }}
        
        .upload-area::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            animation: stageLights 2s linear infinite;
        }}
        
        @keyframes stageLights {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
        
        .upload-area:hover {{
            border-color: #ffeb3b;
            background: linear-gradient(135deg, rgba(255, 182, 193, 0.8) 0%, rgba(238, 130, 238, 0.8) 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: micBounce 1.5s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes micBounce {{
            0%, 100% {{ transform: translateY(0) scale(1); }}
            50% {{ transform: translateY(-10px) scale(1.1); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(255, 105, 180, 0.4);
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
            box-shadow: 0 6px 20px rgba(255, 105, 180, 0.4);
            background: linear-gradient(135deg, #ff69b4 0%, #da70d6 100%);
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
            box-shadow: 0 8px 25px rgba(255, 105, 180, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(255, 192, 203, 0.8) 0%, rgba(221, 160, 221, 0.8) 100%);
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
        
        .video-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(255, 105, 180, 0.2);
            border-left: 5px solid #ff69b4;
            transition: all 0.3s;
        }}
        
        .video-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(255, 105, 180, 0.3);
        }}
        
        .video-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .video-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .video-score {{
            font-size: 1.1em;
            color: #ff69b4;
            font-weight: bold;
        }}
        
        .idol-star {{
            display: inline-block;
            animation: starTwinkle 1s ease-in-out infinite;
        }}
        
        @keyframes starTwinkle {{
            0%, 100% {{ opacity: 1; transform: scale(1) rotate(0deg); }}
            50% {{ opacity: 0.6; transform: scale(1.2) rotate(180deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 视频分类助手</h1>
        <p class="subtitle">偶像少女帮你识别视频内容！支持视频和图片</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🎥</div>
            <p style="font-size: 1.2em; color: #fff; font-weight: bold; position: relative; z-index: 1;">
                点击上传视频或图片~
            </p>
            <p style="color: #ffe; margin-top: 10px; position: relative; z-index: 1;">支持 MP4、AVI、MOV 视频格式</p>
            <p style="color: #ffe; margin-top: 5px; position: relative; z-index: 1;">也支持 JPG、PNG 图片格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="video/*,image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image" style="display: none;">
            <video id="previewVideo" class="preview-image" controls style="display: none;"></video>
        </div>
        
        <button id="classifyBtn" onclick="classifyVideo()" style="display: none;">
            <span class="idol-star">⭐</span> 开始分类 <span class="idol-star">⭐</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建星光闪耀（持续飘落）
        function createStarSparkle() {{
            const stars = ['⭐', '✨', '💫', '🌟', '🎵', '🎶', '♪', '♫'];
            const star = document.createElement('div');
            star.className = 'star-sparkle';
            star.textContent = stars[Math.floor(Math.random() * stars.length)];
            star.style.left = Math.random() * 100 + '%';
            star.style.animationDuration = (Math.random() * 3 + 3) + 's';
            star.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(star);
            
            setTimeout(() => star.remove(), 6000);
        }}
        
        // 每250ms创建一个新星星
        setInterval(createStarSparkle, 250);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                const isVideo = file.type.startsWith('video/');
                
                reader.onload = function(e) {{
                    const previewImage = document.getElementById('previewImage');
                    const previewVideo = document.getElementById('previewVideo');
                    
                    if (isVideo) {{
                        // 显示视频预览
                        previewVideo.src = e.target.result;
                        previewVideo.style.display = 'block';
                        previewImage.style.display = 'none';
                    }} else {{
                        // 显示图片预览
                        previewImage.src = e.target.result;
                        previewImage.style.display = 'block';
                        previewVideo.style.display = 'none';
                    }}
                    
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('classifyBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function classifyVideo() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const classifyBtn = document.getElementById('classifyBtn');
            
            // 判断文件类型
            const isVideo = selectedFile.type.startsWith('video/');
            const fileType = isVideo ? '视频' : '图片';
            
            resultDiv.innerHTML = `<p style="text-align: center; color: #fff; font-size: 1.2em;">🎤 偶像少女正在分析${{fileType}}...</p>`;
            resultDiv.style.display = 'block';
            classifyBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {{
                const response = await fetch('/classify', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ 分类失败: ${{error.message}}</p>`;
            }} finally {{
                classifyBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            const fileType = data.is_video ? '视频' : '图片';
            let html = '<h3 style="color: #fff; margin-bottom: 20px; text-align: center;">🎬 视频分类结果</h3>';
            html += `<p style="text-align: center; color: #fff; margin-bottom: 20px;">
                文件类型: ${{fileType}} | 提取帧数: ${{data.num_frames}} | 识别到 ${{data.predictions.length}} 个可能的类别
            </p>`;
            
            data.predictions.forEach((item, index) => {{
                const labelText = item.label_zh ? `${{item.label_zh}} (${{item.label}})` : item.label;
                html += `
                    <div class="video-item">
                        <div class="video-icon">🎬</div>
                        <div class="video-label">${{labelText}}</div>
                        <div class="video-score">${{(item.score * 100).toFixed(1)}}%</div>
                    </div>
                `;
            }});
            
            if (data.is_video) {{
                html += `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                        <p><strong>🎥 视频处理说明：</strong></p>
                        <p style="margin-top: 5px;">• 系统从视频中均匀提取了16帧进行分析</p>
                        <p>• 分类结果基于整个视频的内容</p>
                        <p>• 视频越清晰，识别效果越好</p>
                    </div>
                `;
            }} else {{
                html += `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                        <p><strong>📸 图片处理说明：</strong></p>
                        <p style="margin-top: 5px;">• 图片被复制为16帧进行视频分类</p>
                        <p>• 适合识别动作、场景等内容</p>
                        <p>• 建议上传动作场景或运动画面</p>
                    </div>
                `;
            }}
            
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

def extract_video_frames(video_path, num_frames=16, target_size=(224, 224)):
    """从视频中提取指定数量的帧，并resize到统一大小"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError("无法读取视频文件")
    
    # 均匀采样帧
    if total_frames < num_frames:
        # 如果视频帧数少于需要的帧数，重复最后一帧
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV读取的是BGR格式，转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image并resize到目标大小
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize(target_size, Image.BILINEAR)
            frames.append(pil_image)
        else:
            # 如果读取失败，使用黑色图像
            frames.append(Image.new('RGB', target_size, (0, 0, 0)))
    
    cap.release()
    
    # 确保返回正确数量的帧
    if len(frames) != num_frames:
        print(f"警告: 期望 {num_frames} 帧，实际获得 {len(frames)} 帧")
        # 补齐或截断
        if len(frames) < num_frames:
            frames.extend([Image.new('RGB', target_size, (0, 0, 0))] * (num_frames - len(frames)))
        else:
            frames = frames[:num_frames]
    
    return frames

@app.route('/classify', methods=['POST'])
def classify():
    temp_file = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        file_content = file.read()
        
        # 判断文件类型
        file_type = file.content_type
        is_video = file_type and file_type.startswith('video/')
        
        if is_video:
            # 处理视频文件
            print(f"处理视频文件: {file.filename}, 类型: {file_type}")
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # 提取视频帧
            frames = extract_video_frames(temp_path, num_frames=16)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            print(f"成功提取 {len(frames)} 帧，每帧大小: {frames[0].size}")
        else:
            # 处理图片文件
            print(f"处理图片文件: {file.filename}")
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
            # Resize到224x224
            image = image.resize((224, 224), Image.BILINEAR)
            # 将单张图片复制为16帧
            frames = [image.copy() for _ in range(16)]
            print(f"图片已复制为16帧，每帧大小: {frames[0].size}")
        
        # 使用模型进行分类
        # VideoMAE处理器直接接受帧列表
        inputs = processor(frames, return_tensors="pt")
        
        # 将输入移到设备上
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 获取top-k结果
        num_classes = logits.shape[-1]
        k = min(5, num_classes)
        top_probs, top_indices = torch.topk(probs, k)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = model.config.id2label.get(idx.item(), f"类别_{idx.item()}")
            
            # 尝试翻译成中文
            label_zh = None
            try:
                translated = translator.translate(label, src='en', dest='zh-cn')
                label_zh = translated.text
                print(f"翻译: {label} -> {label_zh}")
            except Exception as e:
                print(f"翻译失败: {label}, 错误: {e}")
                label_zh = None
            
            predictions.append({
                'label': label,
                'label_zh': label_zh,
                'score': prob.item()
            })
        
        return jsonify({
            'predictions': predictions,
            'is_video': is_video,
            'num_frames': len(frames)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎤 启动偶像少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6007")
    print("🌟 偶像少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6007')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6007, debug=False)
