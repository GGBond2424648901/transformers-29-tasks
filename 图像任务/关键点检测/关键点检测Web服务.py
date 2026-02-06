#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键点检测 Web 服务 - 精灵少女风格 🔍
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image, ImageDraw
import io
import base64
from googletrans import Translator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🔍 关键点检测 Web 服务 - 精灵少女")
print("=" * 70)

print("\n🧚 正在召唤精灵少女...")
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=0)
translator = Translator()
print("🌿 精灵少女准备完毕！开始寻找关键点~")

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
    <title>🔍 关键点检测 - 精灵少女</title>
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
        
        /* 叶子飘落动画 */
        .falling-leaf {{
            position: fixed;
            font-size: 25px;
            animation: leafFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 5px rgba(34,139,34,0.6));
        }}
        
        @keyframes leafFall {{
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
            background: linear-gradient(135deg, rgba(144, 238, 144, 0.95) 0%, rgba(60, 179, 113, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(34, 139, 34, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(34, 139, 34, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .fairy-icon {{
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 80px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            animation: fairyFloat 3s ease-in-out infinite;
        }}
        
        @keyframes fairyFloat {{
            0%, 100% {{ transform: translateX(-50%) translateY(0) rotate(-5deg); }}
            50% {{ transform: translateX(-50%) translateY(-15px) rotate(5deg); }}
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #228b22, #32cd32);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            animation: natureGlow 3s ease-in-out infinite;
        }}
        
        @keyframes natureGlow {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.2); }}
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
            background: linear-gradient(135deg, rgba(152, 251, 152, 0.8) 0%, rgba(60, 179, 113, 0.8) 100%);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }}
        
        .upload-area::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            animation: natureSweep 3s linear infinite;
        }}
        
        @keyframes natureSweep {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .upload-area:hover {{
            border-color: #ffeb3b;
            background: linear-gradient(135deg, rgba(173, 255, 47, 0.8) 0%, rgba(60, 179, 113, 0.8) 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: leafSway 2s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes leafSway {{
            0%, 100% {{ transform: rotate(-10deg); }}
            50% {{ transform: rotate(10deg); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(34, 139, 34, 0.4);
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
            box-shadow: 0 6px 20px rgba(34, 139, 34, 0.4);
            background: linear-gradient(135deg, #228b22 0%, #32cd32 100%);
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
            box-shadow: 0 8px 25px rgba(34, 139, 34, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(152, 251, 152, 0.8) 0%, rgba(60, 179, 113, 0.8) 100%);
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
        
        .keypoint-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(34, 139, 34, 0.2);
            border-left: 5px solid #228b22;
            transition: all 0.3s;
        }}
        
        .keypoint-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(34, 139, 34, 0.3);
        }}
        
        .keypoint-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .keypoint-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .keypoint-score {{
            font-size: 1.1em;
            color: #228b22;
            font-weight: bold;
        }}
        
        .nature-sparkle {{
            display: inline-block;
            animation: sparkle 1.5s ease-in-out infinite;
        }}
        
        @keyframes sparkle {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.6; transform: scale(0.9); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 关键点检测助手</h1>
        <p class="subtitle">精灵少女帮你找到图片中的重要关键点！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🍃</div>
            <p style="font-size: 1.2em; color: #fff; font-weight: bold; position: relative; z-index: 1;">
                点击上传图片寻找关键点~
            </p>
            <p style="color: #ffe; margin-top: 10px; position: relative; z-index: 1;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="detectBtn" onclick="detectKeypoints()" style="display: none;">
            <span class="nature-sparkle">🌿</span> 开始检测 <span class="nature-sparkle">🌿</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建飘落的叶子（持续飘落）
        function createFallingLeaf() {{
            const leaves = ['🍃', '🍂', '🌿', '🌱', '🍀'];
            const leaf = document.createElement('div');
            leaf.className = 'falling-leaf';
            leaf.textContent = leaves[Math.floor(Math.random() * leaves.length)];
            leaf.style.left = Math.random() * 100 + '%';
            leaf.style.animationDuration = (Math.random() * 3 + 4) + 's';
            leaf.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(leaf);
            
            setTimeout(() => leaf.remove(), 7000);
        }}
        
        // 每300ms创建一个新叶子
        setInterval(createFallingLeaf, 300);
        
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
        
        async function detectKeypoints() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const detectBtn = document.getElementById('detectBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #fff; font-size: 1.2em;">🌿 精灵少女正在寻找关键点...</p>';
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
                    resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ 检测失败: ${{error.message}}</p>`;
            }} finally {{
                detectBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            let html = '<h3 style="color: #fff; margin-bottom: 20px; text-align: center;">🔍 关键点检测结果</h3>';
            html += `<p style="text-align: center; color: #fff; margin-bottom: 20px;">共检测到 ${{data.keypoints.length}} 个关键对象</p>`;
            
            if (data.annotated_image) {{
                html += `
                    <div style="text-align: center; margin-bottom: 20px;">
                        <img src="data:image/png;base64,${{data.annotated_image}}" 
                             style="max-width: 100%; border-radius: 15px; border: 3px solid #fff; box-shadow: 0 0 20px rgba(34,139,34,0.5);">
                    </div>
                `;
            }}
            
            data.keypoints.forEach((item, index) => {{
                const labelText = item.label_zh ? `${{item.label_zh}} (${{item.label}})` : item.label;
                html += `
                    <div class="keypoint-item">
                        <div class="keypoint-icon">🎯</div>
                        <div class="keypoint-label">${{labelText}}</div>
                        <div class="keypoint-score">${{(item.score * 100).toFixed(1)}}%</div>
                    </div>
                `;
            }});
            
            html += `
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                    <p><strong>🧚 精灵提示：</strong></p>
                    <p style="margin-top: 5px;">• 图像上的绿色框标注了检测到的关键对象</p>
                    <p>• 中文翻译由Google翻译提供</p>
                    <p>• 括号内为英文原文</p>
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
        
        results = detector(image)
        
        # 翻译标签并准备返回数据
        translated_results = []
        for item in results:
            label = item['label']
            
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
                'score': item['score'],
                'box': item['box']
            })
        
        # 在图片上标注关键点（使用中文标签）
        draw = ImageDraw.Draw(image)
        for item in translated_results:
            box = item['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            
            # 优先使用中文标签
            label_text = item['label_zh'] if item['label_zh'] else item['label']
            draw.text((x1, y1-10), f"{label_text}: {item['score']:.2f}", fill='green')
        
        # 转换标注后的图片为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'keypoints': translated_results,
            'annotated_image': annotated_base64
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
    print("🧚 启动精灵少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6006")
    print("🌿 精灵少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6006')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6006, debug=False)
