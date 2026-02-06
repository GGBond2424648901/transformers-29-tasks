#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度估计 Web 服务 - 科技少女风格 📏
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io
import base64
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("📏 深度估计 Web 服务 - 科技少女")
print("=" * 70)

print("\n🔬 正在召唤科技少女...")
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=0)
print("💻 科技少女准备完毕！开始分析深度~")

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
    <title>📏 深度估计 - 科技少女</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Consolas', monospace;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* 数字雨飘落动画 */
        .digital-rain {{
            position: fixed;
            font-family: 'Consolas', monospace;
            font-size: 20px;
            color: #00ffff;
            animation: digitalFall linear infinite;
            z-index: 1;
            pointer-events: none;
            text-shadow: 0 0 10px #00ffff;
            font-weight: bold;
        }}
        
        @keyframes digitalFall {{
            0% {{
                transform: translateY(-10px);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(0, 20, 40, 0.95) 0%, rgba(0, 40, 60, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 255, 255, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(0, 255, 255, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .tech-icon {{
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 80px;
            filter: drop-shadow(0 4px 8px rgba(0,255,255,0.5));
            animation: techPulse 2s ease-in-out infinite;
        }}
        
        @keyframes techPulse {{
            0%, 100% {{ transform: translateX(-50%) scale(1); filter: drop-shadow(0 4px 8px rgba(0,255,255,0.5)); }}
            50% {{ transform: translateX(-50%) scale(1.1); filter: drop-shadow(0 8px 16px rgba(0,255,255,0.8)); }}
        }}
        
        h1 {{
            text-align: center;
            color: #00ffff;
            margin-bottom: 10px;
            font-size: 2.8em;
            text-shadow: 0 0 20px rgba(0,255,255,0.8);
            animation: techGlow 2s ease-in-out infinite;
        }}
        
        @keyframes techGlow {{
            0%, 100% {{ text-shadow: 0 0 20px rgba(0,255,255,0.8); }}
            50% {{ text-shadow: 0 0 30px rgba(0,255,255,1); }}
        }}
        
        .subtitle {{
            text-align: center;
            color: #66ffff;
            margin-bottom: 30px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .upload-area {{
            border: 3px dashed #00ffff;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, rgba(0, 40, 60, 0.8) 0%, rgba(0, 60, 80, 0.8) 100%);
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
            background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.1), transparent);
            animation: techScan 3s linear infinite;
        }}
        
        @keyframes techScan {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .upload-area:hover {{
            border-color: #66ffff;
            background: linear-gradient(135deg, rgba(0, 60, 80, 0.8) 0%, rgba(0, 80, 100, 0.8) 100%);
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: techRotate 4s linear infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes techRotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 255, 255, 0.4);
            border: 4px solid #00ffff;
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
            box-shadow: 0 6px 20px rgba(0, 255, 255, 0.4);
            background: linear-gradient(135deg, #00aacc 0%, #00ffff 100%);
            color: #001a2e;
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
            box-shadow: 0 8px 25px rgba(0, 255, 255, 0.6);
        }}
        
        button:disabled {{
            background: #444;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(0, 40, 60, 0.8) 0%, rgba(0, 60, 80, 0.8) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #00ffff;
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
        
        .depth-info {{
            background: rgba(0, 255, 255, 0.1);
            padding: 20px;
            margin: 15px 0;
            border-radius: 15px;
            border-left: 5px solid #00ffff;
            color: #66ffff;
            font-size: 1.1em;
        }}
        
        .depth-info strong {{
            color: #00ffff;
        }}
        
        .tech-grid {{
            display: inline-block;
            animation: gridBlink 1.5s ease-in-out infinite;
        }}
        
        @keyframes gridBlink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📏 深度估计助手</h1>
        <p class="subtitle">科技少女帮你分析图片的深度信息！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">💻</div>
            <p style="font-size: 1.2em; color: #00ffff; font-weight: bold; position: relative; z-index: 1;">
                点击上传图片开始深度分析~
            </p>
            <p style="color: #66ffff; margin-top: 10px; position: relative; z-index: 1;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="estimateBtn" onclick="estimateDepth()" style="display: none;">
            <span class="tech-grid">▦</span> 开始估计 <span class="tech-grid">▦</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建数字雨（持续飘落）
        function createDigitalRain() {{
            const chars = ['0', '1', '█', '▓', '▒', '░', '▪', '▫'];
            const rain = document.createElement('div');
            rain.className = 'digital-rain';
            rain.textContent = chars[Math.floor(Math.random() * chars.length)];
            rain.style.left = Math.random() * 100 + '%';
            rain.style.animationDuration = (Math.random() * 2 + 3) + 's';
            rain.style.fontSize = (Math.random() * 10 + 15) + 'px';
            document.body.appendChild(rain);
            
            setTimeout(() => rain.remove(), 5000);
        }}
        
        // 每200ms创建一个新数字
        setInterval(createDigitalRain, 200);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('estimateBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function estimateDepth() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const estimateBtn = document.getElementById('estimateBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #00ffff; font-size: 1.2em;">💻 科技少女正在分析深度数据...</p>';
            resultDiv.style.display = 'block';
            estimateBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {{
                const response = await fetch('/estimate', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #ff6666;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #ff6666;">❌ 估计失败: ${{error.message}}</p>`;
            }} finally {{
                estimateBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            let html = '<h3 style="color: #00ffff; margin-bottom: 20px; text-align: center;">📏 深度分析结果</h3>';
            
            html += `
                <div class="depth-info">
                    <p><strong>🔬 深度图已生成</strong></p>
                    <p style="margin-top: 10px;">深度图显示了图像中每个像素的相对距离</p>
                    <p style="margin-top: 5px;">较亮的区域表示距离较近，较暗的区域表示距离较远</p>
                </div>
            `;
            
            html += `
                <div style="text-align: center; margin-top: 20px;">
                    <img src="data:image/png;base64,${{data.depth_map}}" 
                         style="max-width: 100%; border-radius: 15px; border: 3px solid #00ffff; box-shadow: 0 0 20px rgba(0,255,255,0.5);">
                </div>
            `;
            
            html += `
                <div class="depth-info" style="margin-top: 20px;">
                    <p><strong>💡 提示</strong></p>
                    <p style="margin-top: 10px;">• 深度估计可用于3D重建、AR应用等</p>
                    <p>• 图像质量越好，深度估计越准确</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#66ffff';
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.style.borderColor = '#00ffff';
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#00ffff';
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

@app.route('/estimate', methods=['POST'])
def estimate():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        result = depth_estimator(image)
        depth_map = result['depth']
        
        # 转换深度图为可视化图像
        depth_array = np.array(depth_map)
        depth_normalized = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_normalized)
        
        # 转换为base64
        buffered = io.BytesIO()
        depth_image.save(buffered, format="PNG")
        depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'depth_map': depth_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🔬 启动科技少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6004")
    print("💻 科技少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6004')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6004, debug=False)
