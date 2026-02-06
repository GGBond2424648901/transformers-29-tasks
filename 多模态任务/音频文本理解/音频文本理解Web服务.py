#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文本理解 Web 服务 - 音频解析师 🎵
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

# 添加 ffmpeg 到 PATH
ffmpeg_path = r'D:\transformers训练\transformers-main\预训练模型下载处\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin'
if ffmpeg_path not in os.environ['PATH']:
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
    print(f"✅ 已添加 ffmpeg 到 PATH: {ffmpeg_path}")

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import pipeline
import base64

# 导入翻译库
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
    translator = Translator()
    print("✅ Google翻译支持已启用")
except ImportError:
    TRANSLATOR_AVAILABLE = False
    translator = None
    print("⚠️  未安装 googletrans，翻译功能不可用")

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🎵 音频文本理解 Web 服务 - 音频解析师")
print("=" * 70)

print("\n🎙️ 正在加载语音识别模型...")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
print("✅ 使用 whisper-base 模型（支持中文识别）")

print("\n📝 正在加载中文情感分析模型...")
# 使用已训练好的中文情感分析模型
try:
    classifier = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")
    print("✅ 使用 RoBERTa 中文情感分析模型")
except:
    # 如果上面的模型加载失败，使用备用方案
    print("⚠️  使用基础情感分析（需要训练）")
    classifier = pipeline("text-classification", model="bert-base-chinese")

print("✅ 音频解析师准备完毕！")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 音频文本理解 - 音频解析师</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background-image: url('/static/background');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        .falling-item {
            position: fixed;
            font-size: 29px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.74;
        }
        
        @keyframes fall {
            0% {
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.74;
            }
            100% {
                transform: translateY(100vh) rotate(360deg) scale(1.29);
                opacity: 0.24;
            }
        }
        
        .container {
            background: linear-gradient(135deg, rgba(121, 85, 72, 0.95) 0%, rgba(93, 64, 55, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(121, 85, 72, 0.5);
            padding: 40px;
            max-width: 1000px;
            margin: 20px auto;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(121, 85, 72, 0.6);
            position: relative;
            z-index: 10;
        }
        
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            text-align: center;
            color: #d7ccc8;
            margin-bottom: 30px;
            font-size: 1.2em;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }
        
        .upload-area {
            border: 3px dashed #795548;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(121, 85, 72, 0.05);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: rgba(121, 85, 72, 0.1);
            border-color: #5d4037;
        }
        
        .upload-icon {
            font-size: 3.5em;
            margin-bottom: 15px;
        }
        
        #fileInput {
            display: none;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #795548 0%, #5d4037 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(121, 85, 72, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(215, 204, 200, 0.95) 0%, rgba(188, 170, 164, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #795548;
        }
        
        .result-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #795548;
        }
        
        .result-title {
            color: #5d4037;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .result-content {
            color: #3e2723;
            font-size: 1.1em;
            line-height: 1.8;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(121, 85, 72, 0.1);
            border-left-color: #795548;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频文本理解</h1>
        <p class="subtitle">音频解析师帮你理解音频内容！</p>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">🎙️</div>
                <div>点击或拖拽音频文件到这里</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">支持 MP3、WAV 等格式</div>
                <input type="file" id="fileInput" accept="audio/*">
            </div>
            
            <div class="button-group" id="buttonGroup" style="display: none;">
                <button class="btn-primary" id="analyzeBtn" onclick="analyzeAudio()">🔍 分析音频</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="color: #5d4037; font-size: 1.1em; font-weight: 600;">AI 正在分析音频...</div>
        </div>
        
        <div class="result-container" id="resultContainer"></div>
    </div>

    <script>
        const fallingItems = ['🎵', '🎶', '🎙️', '🎧', '🎤', '🔊', '📻', '🎼', '🎹', '🎸', '✨', '⭐', '🌟', '💫', '🎪', '🎭'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 19 + 23) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        setInterval(createFallingItem, 150);
        
        let selectedFile = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(121, 85, 72, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(121, 85, 72, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(121, 85, 72, 0.05)';
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file || !file.type.startsWith('audio/')) {
                alert('请选择音频文件！');
                return;
            }
            
            selectedFile = file;
            document.getElementById('buttonGroup').style.display = 'flex';
            document.getElementById('resultContainer').style.display = 'none';
            uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <div>已选择: ${file.name}</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">点击重新选择</div>
            `;
        }
        
        async function analyzeAudio() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('resultContainer');
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResult(data);
                } else {
                    alert('分析失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                loading.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            const container = document.getElementById('resultContainer');
            
            let html = '<h3 style="color: #5d4037; margin-bottom: 20px; text-align: center;">✨ 分析结果</h3>';
            
            html += '<div class="result-box">';
            html += '<div class="result-title">🎙️ 语音识别结果：</div>';
            html += `<div class="result-content">${data.transcription}</div>`;
            html += '</div>';
            
            if (data.classification) {
                html += '<div class="result-box">';
                html += '<div class="result-title">📊 情感分析：</div>';
                const label = data.classification.label_cn || '积极';
                const score = (data.classification.score * 100).toFixed(1);
                html += `<div class="result-content">情感: ${label} (置信度: ${score}%)</div>`;
                html += '</div>';
            }
            
            container.innerHTML = html;
            container.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/static/background')
def background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    else:
        return '', 404

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有上传音频'})
        
        file = request.files['audio']
        
        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # 语音识别 - 强制使用中文
        # generate_kwargs 可以指定语言，提高中文识别准确率
        transcription_result = asr(tmp_path, generate_kwargs={"language": "chinese"})
        transcription = transcription_result['text']
        transcription_cn = transcription  # 默认使用原文
        
        # 如果识别结果是英文，翻译成中文
        if TRANSLATOR_AVAILABLE:
            try:
                # 检测语言
                detected = translator.detect(transcription)
                print(f"检测到语言: {detected.lang}")
                
                # 如果不是中文，翻译成中文
                if detected.lang != 'zh-cn' and detected.lang != 'zh':
                    print(f"原文: {transcription}")
                    translated = translator.translate(transcription, src='auto', dest='zh-cn')
                    transcription_cn = translated.text
                    print(f"翻译: {transcription_cn}")
            except Exception as e:
                print(f"翻译失败: {e}")
                transcription_cn = transcription
        
        # 文本分类（情感分析）- 使用中文文本
        classification = None
        try:
            classification_result = classifier(transcription_cn)
            classification = classification_result[0]
            
            # 翻译情感标签为中文
            sentiment_map = {
                'positive': '积极',
                'negative': '消极',
                'neutral': '中性',
                'POSITIVE': '积极',
                'NEGATIVE': '消极',
                'NEUTRAL': '中性',
                'LABEL_0': '消极',
                'LABEL_1': '积极',
                'LABEL_2': '中性',
            }
            
            if 'label' in classification:
                original_label = classification['label']
                # 清理标签文本，只保留主要情感词
                clean_label = original_label.split('(')[0].strip()
                classification['label_cn'] = sentiment_map.get(clean_label, sentiment_map.get(original_label, '积极'))
                classification['label_en'] = original_label
        except Exception as e:
            print(f"情感分析失败: {e}")
            pass
        
        # 删除临时文件
        os.unlink(tmp_path)
        
        return jsonify({
            'success': True,
            'transcription': transcription_cn,  # 返回中文翻译
            'transcription_original': transcription,  # 保留原文（可选）
            'classification': classification
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎵 启动音频解析师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:8004")
    print("🎙️ 音频解析师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:8004')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=8004, debug=False)
