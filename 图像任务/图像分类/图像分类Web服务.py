#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类 Web 服务 - 可爱猫娘风格 🐱
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io
import base64
from googletrans import Translator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🐱 图像分类 Web 服务 - 猫娘助手")
print("=" * 70)

# 加载模型
print("\n🎀 正在召唤猫娘助手...")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=0)
translator = Translator()
print("✨ 猫娘助手准备完毕！喵~")

app = Flask(__name__)

# 读取背景图片
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
    <title>🐱 图像分类 - 猫娘助手</title>
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
        
        /* 樱花飘落动画 */
        .sakura {{
            position: fixed;
            top: -10px;
            font-size: 20px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
        }}
        
        @keyframes fall {{
            to {{
                transform: translateY(100vh) rotate(360deg);
            }}
        }}
        
        .container {{
            background: rgba(255, 240, 245, 0.95);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 105, 180, 0.4);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 182, 193, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .cat-ears {{
            position: absolute;
            top: -30px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 60px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #ff69b4, #ff1493);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            text-shadow: 2px 2px 4px rgba(255,105,180,0.3);
            animation: bounce 2s ease-in-out infinite;
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        .subtitle {{
            text-align: center;
            color: #ff69b4;
            margin-bottom: 30px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .upload-area {{
            border: 3px dashed #ffb6c1;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
        }}
        
        .upload-area:hover {{
            border-color: #ff69b4;
            background: linear-gradient(135deg, #ffe4e1 0%, #ffc0cb 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: float 3s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-15px); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(255,105,180,0.3);
            border: 4px solid #ffb6c1;
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
            box-shadow: 0 6px 20px rgba(255,105,180,0.4);
            background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
            color: white;
            margin-bottom: 15px;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255,105,180,0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #ffb6c1;
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
        
        .result-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(255,105,180,0.2);
            border-left: 5px solid #ff69b4;
        }}
        
        .result-rank {{
            font-size: 2em;
            font-weight: bold;
            color: #ff69b4;
            margin-right: 20px;
            min-width: 50px;
        }}
        
        .result-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .result-score {{
            font-size: 1.1em;
            color: #ff1493;
            font-weight: bold;
        }}
        
        .paw-print {{
            display: inline-block;
            animation: rotate 2s linear infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐱 图像分类助手</h1>
        <p class="subtitle">喵~ 让猫娘帮你识别图片吧！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🎀</div>
            <p style="font-size: 1.2em; color: #ff69b4; font-weight: bold;">
                点击上传图片或拖拽到这里喵~
            </p>
            <p style="color: #999; margin-top: 10px;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="classifyBtn" onclick="classifyImage()" style="display: none;">
            <span class="paw-print">🐾</span> 开始识别 <span class="paw-print">🐾</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建樱花
        function createSakura() {{
            const sakura = document.createElement('div');
            sakura.className = 'sakura';
            sakura.textContent = '🌸';
            sakura.style.left = Math.random() * 100 + '%';
            sakura.style.animationDuration = (Math.random() * 3 + 5) + 's';
            sakura.style.opacity = Math.random() * 0.5 + 0.3;
            document.body.appendChild(sakura);
            
            setTimeout(() => sakura.remove(), 8000);
        }}
        
        setInterval(createSakura, 300);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('classifyBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function classifyImage() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const classifyBtn = document.getElementById('classifyBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #ff69b4; font-size: 1.2em;">🐱 猫娘正在努力识别中... 喵~</p>';
            resultDiv.style.display = 'block';
            classifyBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {{
                const response = await fetch('/classify', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #ff1493;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data.results);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #ff1493;">❌ 识别失败: ${{error.message}}</p>`;
            }} finally {{
                classifyBtn.disabled = false;
            }}
        }}
        
        function displayResults(results) {{
            let html = '<h3 style="color: #ff69b4; margin-bottom: 20px; text-align: center;">✨ 识别结果 ✨</h3>';
            
            results.forEach((item, index) => {{
                const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🏅';
                const labelText = item.label_zh ? `${{item.label_zh}} (${{item.label}})` : item.label;
                html += `
                    <div class="result-item">
                        <div class="result-rank">${{medal}}</div>
                        <div class="result-label">${{labelText}}</div>
                        <div class="result-score">${{(item.score * 100).toFixed(2)}}%</div>
                    </div>
                `;
            }});
            
            html += `
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                    <p><strong>🐱 猫娘提示：</strong></p>
                    <p style="margin-top: 5px;">• 中文翻译由Google翻译提供</p>
                    <p>• 括号内为英文原文</p>
                    <p>• 上传清晰的图片可以获得更好的识别效果喵~</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        // 拖拽上传
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#ff69b4';
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.style.borderColor = '#ffb6c1';
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#ffb6c1';
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

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片喵~'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        results = classifier(image, top_k=5)
        
        # 添加中文翻译
        translated_results = []
        for result in results:
            label = result['label']
            score = result['score']
            
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
                'score': score
            })
        
        return jsonify({'results': translated_results})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🌸 启动猫娘助手...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6001")
    print("💕 猫娘在这里等你哦~ 喵~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6001')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6001, debug=False)
