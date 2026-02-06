#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分割 Web 服务 - 魔法少女风格 🎭
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
from googletrans import Translator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🎭 图像分割 Web 服务 - 魔法少女")
print("=" * 70)

print("\n✨ 正在召唤魔法少女...")
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device=0)
translator = Translator()
print("🌟 魔法少女准备完毕！开始施展魔法~")

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
    <title>🎭 图像分割 - 魔法少女</title>
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
        
        /* 魔法粒子飘落动画 */
        .magic-particle {{
            position: fixed;
            font-size: 25px;
            animation: magicFall linear infinite;
            z-index: 1;
            pointer-events: none;
            filter: drop-shadow(0 0 5px rgba(255,255,255,0.8));
        }}
        
        @keyframes magicFall {{
            0% {{
                transform: translateY(-10px) rotate(0deg);
                opacity: 1;
            }}
            100% {{
                transform: translateY(100vh) rotate(720deg);
                opacity: 0.3;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(255, 240, 255, 0.95) 0%, rgba(240, 230, 255, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(155, 89, 182, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(155, 89, 182, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        .magic-wand {{
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 80px;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            animation: wandFloat 3s ease-in-out infinite;
        }}
        
        @keyframes wandFloat {{
            0%, 100% {{ transform: translateX(-50%) translateY(0) rotate(-10deg); }}
            50% {{ transform: translateX(-50%) translateY(-15px) rotate(10deg); }}
        }}
        
        h1 {{
            text-align: center;
            background: linear-gradient(45deg, #9b59b6, #e74c3c, #f39c12);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            animation: rainbow 3s ease-in-out infinite;
        }}
        
        @keyframes rainbow {{
            0%, 100% {{ filter: hue-rotate(0deg); }}
            50% {{ filter: hue-rotate(30deg); }}
        }}
        
        .subtitle {{
            text-align: center;
            color: #9b59b6;
            margin-bottom: 30px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .upload-area {{
            border: 3px dashed #9b59b6;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #ffeef8 0%, #f3e5f5 100%);
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
            background: linear-gradient(45deg, transparent, rgba(155, 89, 182, 0.1), transparent);
            animation: shine 3s linear infinite;
        }}
        
        @keyframes shine {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .upload-area:hover {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #f3e5f5 0%, #fce4ec 100%);
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 60px;
            margin-bottom: 15px;
            animation: sparkle 2s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }}
        
        @keyframes sparkle {{
            0%, 100% {{ transform: scale(1) rotate(0deg); }}
            25% {{ transform: scale(1.1) rotate(-5deg); }}
            75% {{ transform: scale(1.1) rotate(5deg); }}
        }}
        
        .preview-container {{
            margin: 25px 0;
            text-align: center;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(155, 89, 182, 0.4);
            border: 4px solid #9b59b6;
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
            box-shadow: 0 6px 20px rgba(155, 89, 182, 0.4);
            background: linear-gradient(135deg, #9b59b6 0%, #e74c3c 100%);
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
            box-shadow: 0 8px 25px rgba(155, 89, 182, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, #ffeef8 0%, #f3e5f5 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #9b59b6;
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
        
        .segment-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(155, 89, 182, 0.2);
            border-left: 5px solid #9b59b6;
            transition: all 0.3s;
        }}
        
        .segment-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(155, 89, 182, 0.3);
        }}
        
        .segment-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .segment-label {{
            flex: 1;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .segment-score {{
            font-size: 1.1em;
            color: #9b59b6;
            font-weight: bold;
        }}
        
        .magic-star {{
            display: inline-block;
            animation: twinkle 1.5s ease-in-out infinite;
        }}
        
        @keyframes twinkle {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(0.8); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 图像分割助手</h1>
        <p class="subtitle">魔法少女帮你分割图片的每个部分！</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">✨</div>
            <p style="font-size: 1.2em; color: #9b59b6; font-weight: bold; position: relative; z-index: 1;">
                点击上传图片施展魔法~
            </p>
            <p style="color: #999; margin-top: 10px; position: relative; z-index: 1;">支持 JPG、PNG 格式</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        
        <div id="previewContainer" class="preview-container" style="display: none;">
            <img id="previewImage" class="preview-image">
        </div>
        
        <button id="segmentBtn" onclick="segmentImage()" style="display: none;">
            <span class="magic-star">⭐</span> 开始分割 <span class="magic-star">⭐</span>
        </button>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建魔法粒子（持续飘落）
        function createMagicParticle() {{
            const particles = ['✨', '🌟', '💫', '⭐', '🔮'];
            const particle = document.createElement('div');
            particle.className = 'magic-particle';
            particle.textContent = particles[Math.floor(Math.random() * particles.length)];
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDuration = (Math.random() * 3 + 4) + 's';
            particle.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(particle);
            
            setTimeout(() => particle.remove(), 7000);
        }}
        
        // 每300ms创建一个新粒子
        setInterval(createMagicParticle, 300);
        
        let selectedFile = null;
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('segmentBtn').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function segmentImage() {{
            if (!selectedFile) return;
            
            const resultDiv = document.getElementById('result');
            const segmentBtn = document.getElementById('segmentBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #9b59b6; font-size: 1.2em;">✨ 魔法少女正在施展分割魔法...</p>';
            resultDiv.style.display = 'block';
            segmentBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {{
                const response = await fetch('/segment', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #e74c3c;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResults(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #e74c3c;">❌ 分割失败: ${{error.message}}</p>`;
            }} finally {{
                segmentBtn.disabled = false;
            }}
        }}
        
        function displayResults(data) {{
            let html = '<h3 style="color: #9b59b6; margin-bottom: 20px; text-align: center;">🎭 分割结果</h3>';
            
            // 显示分割后的图像
            if (data.segmented_image) {{
                html += `
                    <div style="text-align: center; margin-bottom: 20px;">
                        <img src="data:image/png;base64,${{data.segmented_image}}" 
                             style="max-width: 100%; border-radius: 15px; box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);">
                    </div>
                `;
            }}
            
            html += `<p style="text-align: center; color: #666; margin-bottom: 20px;">共识别到 ${{data.segments.length}} 个区域</p>`;
            
            data.segments.forEach((item, index) => {{
                const labelText = item.label_zh ? `${{item.label_zh}} (${{item.label}})` : item.label;
                const scoreText = item.score > 0 ? `${{(item.score * 100).toFixed(1)}}%` : '检测到';
                
                // 从rgba字符串中提取颜色值
                const colorMatch = item.color.match(/rgba\\((\\d+),\\s*(\\d+),\\s*(\\d+),\\s*([\\d.]+)\\)/);
                let colorStyle = 'background: #9b59b6;';
                if (colorMatch) {{
                    const r = colorMatch[1];
                    const g = colorMatch[2];
                    const b = colorMatch[3];
                    colorStyle = `background: rgb(${{r}}, ${{g}}, ${{b}});`;
                }}
                
                html += `
                    <div class="segment-item">
                        <div style="width: 30px; height: 30px; border-radius: 5px; ${{colorStyle}} margin-right: 10px; border: 2px solid #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.2);"></div>
                        <div class="segment-label">${{labelText}}</div>
                        <div class="segment-score">${{scoreText}}</div>
                    </div>
                `;
            }});
            
            html += `
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #666;">
                    <p><strong>✨ 魔法提示：</strong></p>
                    <p style="margin-top: 5px;">• 彩色方块对应图像中的分割区域颜色</p>
                    <p>• 百分比表示该区域占图像的面积比例</p>
                    <p>• 上传清晰的图片可以获得更好的分割效果</p>
                </div>
            `;
            
            document.getElementById('result').innerHTML = html;
        }}
        
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#e74c3c';
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.style.borderColor = '#9b59b6';
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.style.borderColor = '#9b59b6';
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

@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        results = segmenter(image)
        
        # 创建彩色分割图
        segmented_image = image.copy()
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 为每个区域生成不同的颜色
        colors = [
            (255, 0, 0, 100),    # 红色
            (0, 255, 0, 100),    # 绿色
            (0, 0, 255, 100),    # 蓝色
            (255, 255, 0, 100),  # 黄色
            (255, 0, 255, 100),  # 品红
            (0, 255, 255, 100),  # 青色
            (255, 128, 0, 100),  # 橙色
            (128, 0, 255, 100),  # 紫色
            (255, 192, 203, 100),# 粉色
            (128, 255, 0, 100),  # 黄绿
        ]
        
        # 将结果中的PIL Image对象转换为base64字符串
        segments = []
        total_pixels = image.size[0] * image.size[1]
        
        for idx, result in enumerate(results):
            # 获取mask并应用颜色
            mask_image = result['mask']
            mask_array = np.array(mask_image)
            
            # 计算mask覆盖的像素数量作为score
            mask_pixels = np.sum(mask_array > 0)
            coverage_score = mask_pixels / total_pixels
            
            # 在overlay上绘制彩色区域
            color = colors[idx % len(colors)]
            colored_mask = Image.new('RGBA', image.size, color)
            mask_alpha = Image.fromarray(mask_array).convert('L')
            overlay.paste(colored_mask, (0, 0), mask_alpha)
            
            # 将mask转换为base64
            buffered = io.BytesIO()
            mask_image.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 获取原始score，如果没有则使用coverage_score
            score = result.get('score')
            if score is None or score == 0.0:
                score = coverage_score
            
            label = result.get('label', 'unknown')
            
            # 翻译标签
            label_zh = None
            try:
                translated = translator.translate(label, src='en', dest='zh-cn')
                label_zh = translated.text
                print(f"翻译: {label} -> {label_zh}, 覆盖率: {coverage_score*100:.1f}%")
            except Exception as e:
                print(f"翻译失败: {label}, 错误: {e}")
                label_zh = None
            
            segments.append({
                'label': label,
                'label_zh': label_zh,
                'score': float(score),
                'mask': mask_base64,
                'color': f'rgba{color}'
            })
        
        # 合成最终的分割图像
        segmented_image = Image.alpha_composite(segmented_image.convert('RGBA'), overlay)
        
        # 将分割后的图像转换为base64
        buffered = io.BytesIO()
        segmented_image.convert('RGB').save(buffered, format="PNG")
        segmented_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'segments': segments,
            'segmented_image': segmented_base64
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
    print("🌈 启动魔法少女...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:6003")
    print("✨ 魔法少女在这里等你~\n")
    
    # 延迟1秒后自动打开浏览器
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:6003')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=6003, debug=False)
