#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测 Web 服务 - 侦探少女风格 🎯
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import random
from googletrans import Translator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🎯 目标检测 Web 服务 - 侦探少女")
print("=" * 70)

print("\n🔍 正在召唤侦探少女...")
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=0)
translator = Translator()
print("✨ 侦探少女准备完毕！开始调查~")

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
    <title>🎯 目标检测 - 侦探少女</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Courier New', monospace;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* 星星飘落动画 */
        .falling-star {{
            position: fixed;
            font-size: 20px;
            animation: starFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 5px rgba(65,105,225,0.8));
        }}
        
        @keyframes starFall {{
            0% {{
                transform: translateY(-10px) rotate(0deg);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg);
                opacity: 0.3;
            }}
        }}
        
        .container {{
            background: rgba(230, 240, 255, 0.95);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(65, 105, 225, 0.4);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(100, 149, 237, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .detective-hat {{
            position: absolute;
            top: -35px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 70px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            animation: swing 3s ease-in-out infinite;
        }}
        
        @keyframes swing {{
            0%, 100% {{ transform: translateX(-50%) rotate(-5deg); }}
            50% {{ transform: translateX(-50%) rotate(5deg); }}
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #4169e1, #6495ed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            text-shadow: 2px 2px 4px rgba(65,105,225,0.3);
        }}
        
        .subtitle {{
            text-align: center;
            color: #4169e1;
            margin-bottom: 30px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .upload-area {{
            border: 3px dashed #6495ed;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #e6f2ff 0%, #cce5ff 100%);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
        }}
        
        .upload-area:hover {{
            border-color: #4169e1;
            background: linear-gradient(135deg, #cce5ff 0%, #b3d9ff 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: magnify 2s ease-in-out infinite;
        }}
        
        @keyframes magnify {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(65,105,225,0.3);
            border: 4px solid #6495ed;
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
            box-shadow: 0 6px 20px rgba(65,105,225,0.4);
            background: linear-gradient(135deg, #4169e1 0%, #6495ed 100%);
            color: white;
            margin-bottom: 15px;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(65,105,225,0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, #e6f2ff 0%, #cce5ff 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #6495ed;
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
        
        .detection-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(65,105,225,0.2);
            border-left: 5px solid #4169e1;
        }}
        
        .detection-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .detection-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .detection-score {{
            font-size: 1.1em;
            color: #4169e1;
            font-weight: bold;
        }}
        
        .magnifier {{
            display: inline-block;
            animation: rotate 3s linear infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 目标检测助手</h1>
        <p class="subtitle">侦探少女帮你找出图片中的所有物体！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🔍</div>
            <p style="font-size: 1.2em; color: #4169e1; font-weight: bold;">
                点击上传图片开始调查~
            </p>
            <p style="color: #999; margin-top: 10px;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="detectBtn" onclick="detectObjects()" style="display: none;">
            <span class="magnifier">🔎</span> 开始检测 <span class="magnifier">🔎</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建飘落的星星（持续飘落）
        function createFallingStar() {{
            const stars = ['⭐', '✨', '💫', '🌟'];
            const star = document.createElement('div');
            star.className = 'falling-star';
            star.textContent = stars[Math.floor(Math.random() * stars.length)];
            star.style.left = Math.random() * 100 + '%';
            star.style.animationDuration = (Math.random() * 3 + 3) + 's';
            star.style.fontSize = (Math.random() * 10 + 15) + 'px';
            document.body.appendChild(star);
            
            setTimeout(() => star.remove(), 6000);
        }}
        
        // 每250ms创建一个新星星
        setInterval(createFallingStar, 250);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('detectBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function detectObjects() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const detectBtn = document.getElementById('detectBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #4169e1; font-size: 1.2em;">🔍 侦探少女正在调查中...</p>';
            resultDiv.style.display = 'block';
            detectBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {{
                const response = await fetch('/detect', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #4169e1;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #4169e1;">❌ 检测失败: ${{error.message}}</p>`;
            }} finally {{
                detectBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            let html = '<h3 style="color: #4169e1; margin-bottom: 20px; text-align: center;">🎯 检测结果</h3>';
            html += `<p style="text-align: center; color: #666; margin-bottom: 20px;">共检测到 ${{data.detections.length}} 个物体</p>`;
            
            if (data.annotated_image) {{
                html += `<div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,${{data.annotated_image}}" 
                         style="max-width: 100%; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                </div>`;
            }}
            
            data.detections.forEach((item, index) => {{
                const labelText = item.label_zh ? `${{item.label_zh}} (${{item.label}})` : item.label;
                html += `
                    <div class="detection-item">
                        <div class="detection-icon">🎯</div>
                        <div class="detection-label">${{labelText}}</div>
                        <div class="detection-score">${{(item.score * 100).toFixed(1)}}%</div>
                    </div>
                `;
            }});
            
            html += `
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                    <p><strong>🔍 侦探提示：</strong></p>
                    <p style="margin-top: 5px;">• 图像上的框和标签已标注检测到的物体</p>
                    <p>• 中文翻译由Google翻译提供</p>
                    <p>• 不同颜色的框代表不同的检测对象</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#4169e1';
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.style.borderColor = '#6495ed';
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#6495ed';
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
        
        results = detector(image)
        
        # 翻译标签并准备返回数据
        translated_results = []
        for detection in results:
            label = detection['label']
            
            # 翻译标签
            label_zh = None
            try:
                translated = translator.translate(label, src='en', dest='zh-cn')
                label_zh = translated.text
                print(f"翻译: {label} -> {label_zh}")
            except Exception as e:
                print(f"翻译失败: {label}, 错误: {e}")
                label_zh = None
            
            translated_results.append({
                'label': label,
                'label_zh': label_zh,
                'score': detection['score'],
                'box': detection['box']
            })
        
        # 在图片上绘制检测框
        draw = ImageDraw.Draw(image)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        for i, detection in enumerate(translated_results):
            box = detection['box']
            color = colors[i % len(colors)]
            
            # 绘制矩形框
            draw.rectangle(
                [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
                outline=color,
                width=3
            )
            
            # 绘制标签（优先使用中文）
            label_text = detection['label_zh'] if detection['label_zh'] else detection['label']
            label = f"{label_text} {detection['score']:.2f}"
            draw.text((box['xmin'], box['ymin']-20), label, fill=color)
        
        # 转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'detections': translated_results,
            'annotated_image': img_str
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🌟 启动侦探少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6002")
    print("🔍 侦探少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6002')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6002, debug=False)
