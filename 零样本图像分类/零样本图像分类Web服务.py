#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本图像分类 Web 服务 - 魔导书少女风格 🌈
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io
import base64

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🌈 零样本图像分类 Web 服务 - 魔导书少女")
print("=" * 70)

print("\n📖 正在召唤魔导书少女...")
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=0)
print("✨ 魔导书少女准备完毕！可以识别任意类别~")

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
    <title>🌈 零样本图像分类 - 魔导书少女</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Georgia', serif;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* 魔法阵飘落动画 */
        .magic-circle {{
            position: fixed;
            font-size: 30px;
            animation: circleFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 10px rgba(255,215,0,0.8));
        }}
        
        @keyframes circleFall {{
            0% {{
                transform: translateY(-10px) rotate(0deg);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.95) 0%, rgba(255, 165, 0, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 215, 0, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 215, 0, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .book-icon {{
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 80px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            animation: bookFloat 3s ease-in-out infinite;
        }}
        
        @keyframes bookFloat {{
            0%, 100% {{ transform: translateX(-50%) translateY(0) rotateY(0deg); }}
            50% {{ transform: translateX(-50%) translateY(-15px) rotateY(10deg); }}
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #ffd700, #ff8c00, #ff6347);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            animation: magicGlow 3s ease-in-out infinite;
        }}
        
        @keyframes magicGlow {{
            0%, 100% {{ filter: brightness(1) drop-shadow(0 0 10px rgba(255,215,0,0.5)); }}
            50% {{ filter: brightness(1.3) drop-shadow(0 0 20px rgba(255,215,0,0.8)); }}
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
            background: linear-gradient(135deg, rgba(255, 228, 100, 0.8) 0%, rgba(255, 180, 50, 0.8) 100%);
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
            width: 200px;
            height: 200px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: magicRing 3s linear infinite;
        }}
        
        @keyframes magicRing {{
            0% {{
                width: 200px;
                height: 200px;
                opacity: 1;
            }}
            100% {{
                width: 400px;
                height: 400px;
                opacity: 0;
            }}
        }}
        
        .upload-area:hover {{
            border-color: #ff6347;
            background: linear-gradient(135deg, rgba(255, 240, 120, 0.8) 0%, rgba(255, 200, 70, 0.8) 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: bookOpen 2s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes bookOpen {{
            0%, 100% {{ transform: rotateY(0deg); }}
            50% {{ transform: rotateY(20deg); }}
        }}
        
        .labels-input {{
            width: 100%;
            padding: 15px;
            font-size: 1.1em;
            border: 3px solid #fff;
            border-radius: 15px;
            margin-bottom: 20px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-weight: bold;
        }}
        
        .labels-input:focus {{
            outline: none;
            border-color: #ff6347;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
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
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
            background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
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
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(255, 228, 100, 0.8) 0%, rgba(255, 180, 50, 0.8) 100%);
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
        
        .class-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
            border-left: 5px solid #ffd700;
            transition: all 0.3s;
        }}
        
        .class-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.3);
        }}
        
        .class-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .class-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .class-score {{
            font-size: 1.1em;
            color: #ff8c00;
            font-weight: bold;
        }}
        
        .magic-symbol {{
            display: inline-block;
            animation: symbolRotate 2s linear infinite;
        }}
        
        @keyframes symbolRotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .hint {{
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: #666;
            font-size: 0.95em;
            border-left: 4px solid #ffd700;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌈 零样本图像分类</h1>
        <p class="subtitle">魔导书少女可以识别任意你想要的类别！</p>
        
        <div class="hint">
            💡 <strong>提示：</strong>在下方输入你想识别的类别，用<strong>英文逗号</strong>分隔。例如：猫,狗,鸟,汽车,飞机
        </div>
        
        <input type="text" 
               id="labelsInput" 
               class="labels-input" 
               placeholder="输入类别标签，用逗号分隔（例如：猫,狗,鸟）"
               value="猫,狗,鸟,汽车,飞机,树,花,人,建筑,食物">
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📚</div>
            <p style="font-size: 1.2em; color: #fff; font-weight: bold; position: relative; z-index: 1;">
                点击上传图片开始魔法识别~
            </p>
            <p style="color: #ffe; margin-top: 10px; position: relative; z-index: 1;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="classifyBtn" onclick="classifyImage()" style="display: none;">
            <span class="magic-symbol">🔮</span> 开始识别 <span class="magic-symbol">🔮</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建魔法阵（持续飘落）
        function createMagicCircle() {{
            const circles = ['⭕', '🔴', '🟡', '🟠', '⚪', '🔵', '🟣'];
            const circle = document.createElement('div');
            circle.className = 'magic-circle';
            circle.textContent = circles[Math.floor(Math.random() * circles.length)];
            circle.style.left = Math.random() * 100 + '%';
            circle.style.animationDuration = (Math.random() * 3 + 4) + 's';
            circle.style.fontSize = (Math.random() * 15 + 20) + 'px';
            document.body.appendChild(circle);
            
            setTimeout(() => circle.remove(), 7000);
        }}
        
        // 每350ms创建一个新魔法阵
        setInterval(createMagicCircle, 350);
        
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
            
            const labels = document.getElementById('labelsInput').value.trim();
            if (!labels) {{
                alert('请输入至少一个类别标签！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const classifyBtn = document.getElementById('classifyBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #fff; font-size: 1.2em;">📖 魔导书少女正在施展识别魔法...</p>';
            resultDiv.style.display = 'block';
            classifyBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('labels', labels);
            
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
                resultDiv.innerHTML = `<p style="text-align: center; color: #fff;">❌ 识别失败: ${{error.message}}</p>`;
            }} finally {{
                classifyBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            let html = '<h3 style="color: #fff; margin-bottom: 20px; text-align: center;">🌈 识别结果</h3>';
            html += `<p style="text-align: center; color: #fff; margin-bottom: 20px;">在 ${{data.results.length}} 个类别中进行了识别</p>`;
            
            data.results.forEach((item, index) => {{
                html += `
                    <div class="class-item">
                        <div class="class-icon">🏷️</div>
                        <div class="class-label">${{item.label}}</div>
                        <div class="class-score">${{(item.score * 100).toFixed(1)}}%</div>
                    </div>
                `;
            }});
            
            html += `
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                    <p><strong>✨ 魔法提示：</strong></p>
                    <p style="margin-top: 5px;">• 零样本分类无需训练即可识别任意类别</p>
                    <p>• 尝试输入不同的类别标签获得不同结果</p>
                    <p>• 标签越具体，识别效果越好</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#ff6347';
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

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        if 'labels' not in request.form:
            return jsonify({'error': '没有提供类别标签'}), 400
        
        file = request.files['image']
        labels_str = request.form['labels']
        
        # 解析标签 - 支持中英文逗号和分号
        # 先统一替换为英文逗号
        labels_str = labels_str.replace('，', ',').replace('；', ';').replace(';', ',')
        labels = [label.strip() for label in labels_str.split(',') if label.strip()]
        
        if not labels:
            return jsonify({'error': '请至少提供一个类别标签'}), 400
        
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        results = classifier(image, candidate_labels=labels)
        
        # CLIP零样本分类返回的是列表格式
        # 每个元素是字典: {'score': float, 'label': str}
        formatted_results = []
        for item in results:
            formatted_results.append({
                'label': item['label'],
                'score': float(item['score'])
            })
        
        return jsonify({'results': formatted_results})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': f'{str(e)}'}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("📖 启动魔导书少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6008")
    print("✨ 魔导书少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6008')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6008, debug=False)
