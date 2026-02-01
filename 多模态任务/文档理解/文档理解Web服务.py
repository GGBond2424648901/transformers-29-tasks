#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档理解 Web 服务 - 文档解析师 📄
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import pipeline
from PIL import Image
import base64
import io
import tempfile

# 导入PDF和Word处理库
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  未安装 pdf2image，PDF支持不可用。安装: pip install pdf2image")

try:
    from docx2pdf import convert as docx_to_pdf
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️  未安装 docx2pdf，Word支持不可用。安装: pip install docx2pdf")

try:
    from docx import Document
    import pythoncom
    DOCX_READ_SUPPORT = True
except ImportError:
    DOCX_READ_SUPPORT = False
    print("⚠️  未安装 python-docx，Word读取支持不可用。安装: pip install python-docx")

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("📄 文档理解 Web 服务 - 文档解析师")
print("=" * 70)

print("\n📚 正在加载文档问答模型...")
doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
print("✅ 文档解析师准备完毕！")

# 打印支持的格式
print("\n📋 支持的文档格式:")
print("  ✅ 图片格式: JPG, PNG, BMP, GIF")
if PDF_SUPPORT:
    print("  ✅ PDF文档")
else:
    print("  ❌ PDF文档 (需要安装 pdf2image 和 poppler)")
if DOCX_READ_SUPPORT:
    print("  ✅ Word文档 (.docx)")
else:
    print("  ❌ Word文档 (需要安装 python-docx)")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 文档理解 - 文档解析师</title>
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
            font-size: 26px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.7;
        }
        
        @keyframes fall {
            0% {
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.7;
            }
            100% {
                transform: translateY(100vh) rotate(360deg) scale(1.26);
                opacity: 0.24;
            }
        }
        
        .container {
            background: linear-gradient(135deg, rgba(63, 81, 181, 0.95) 0%, rgba(48, 63, 159, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(63, 81, 181, 0.5);
            padding: 40px;
            max-width: 1100px;
            margin: 20px auto;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(63, 81, 181, 0.6);
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
            color: #c5cae9;
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
            border: 3px dashed #3f51b5;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(63, 81, 181, 0.05);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            background: rgba(63, 81, 181, 0.1);
            border-color: #303f9f;
        }
        
        .upload-icon {
            font-size: 3.5em;
            margin-bottom: 15px;
        }
        
        #fileInput {
            display: none;
        }
        
        .preview-container {
            display: none;
            margin-bottom: 20px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: block;
            margin: 0 auto 20px;
        }
        
        .question-area {
            margin-top: 20px;
        }
        
        .question-area label {
            display: block;
            color: #303f9f;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #3f51b5;
            border-radius: 15px;
            font-size: 1.05em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            transition: all 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #303f9f;
            box-shadow: 0 0 15px rgba(63, 81, 181, 0.3);
        }
        
        .quick-questions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 8px 15px;
            background: linear-gradient(135deg, #5c6bc0 0%, #3f51b5 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .quick-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(63, 81, 181, 0.4);
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
            background: linear-gradient(135deg, #3f51b5 0%, #303f9f 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(63, 81, 181, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #5c6bc0 0%, #3949ab 100%);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(92, 107, 192, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(197, 202, 233, 0.95) 0%, rgba(159, 168, 218, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #3f51b5;
        }
        
        .answer-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #3f51b5;
        }
        
        .question-text {
            color: #303f9f;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .answer-text {
            color: #1a237e;
            font-size: 1.3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .confidence {
            color: #666;
            font-size: 0.95em;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e8eaf6;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #3f51b5 0%, #303f9f 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(63, 81, 181, 0.1);
            border-left-color: #3f51b5;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .info-box {
            background: linear-gradient(135deg, #e8eaf615 0%, #c5cae915 100%);
            border-left: 4px solid #3f51b5;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .info-box p {
            color: #303f9f;
            line-height: 1.8;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 文档理解</h1>
        <p class="subtitle">文档解析师帮你理解文档内容！</p>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📄</div>
                <div>点击或拖拽文档到这里</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">支持 PDF、Word、图片等格式</div>
                <input type="file" id="fileInput" accept="image/*,.pdf,.doc,.docx">
            </div>
            
            <div class="preview-container" id="previewContainer">
                <img id="previewImage" class="preview-image" alt="文档预览">
                
                <div class="question-area">
                    <label>❓ 向文档提问：</label>
                    <input type="text" id="questionInput" placeholder="例如：What is the total amount?">
                    <div class="quick-questions">
                        <button class="quick-btn" onclick="setQuestion('What is the invoice number?')">发票号</button>
                        <button class="quick-btn" onclick="setQuestion('What is the date?')">日期</button>
                        <button class="quick-btn" onclick="setQuestion('What is the total amount?')">总金额</button>
                        <button class="quick-btn" onclick="setQuestion('Who is the vendor?')">供应商</button>
                    </div>
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" id="askBtn" onclick="askQuestion()">🔍 提问</button>
                    <button class="btn-secondary" id="changeBtn" onclick="changeDocument()">🔄 更换文档</button>
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="color: #303f9f; font-size: 1.1em; font-weight: 600;">AI 正在分析文档...</div>
        </div>
        
        <div class="result-container" id="resultContainer"></div>
        
        <div class="info-box">
            <p><strong>🤖 模型：</strong>impira/layoutlm-document-qa</p>
            <p><strong>💡 功能：</strong>OCR文字识别 + 布局分析 + 信息提取</p>
            <p><strong>📋 支持格式：</strong>PDF、Word(.docx)、图片(JPG/PNG/BMP)</p>
            <p><strong>📋 应用：</strong>发票处理、合同分析、表单填充、文档自动化</p>
        </div>
    </div>

    <script>
        const fallingItems = ['📄', '📃', '📋', '📑', '📊', '📈', '🔍', '🔎', '✨', '⭐', '🌟', '💫', '📌', '🔖', '💼', '🏢'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 16 + 20) + 'px';
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
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(63, 81, 181, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(63, 81, 181, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(63, 81, 181, 0.05)';
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file) {
                alert('请选择文件！');
                return;
            }
            
            // 检查文件类型
            const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif', 
                              'application/pdf', 
                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                              'application/msword'];
            
            const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.pdf', '.doc', '.docx'];
            const fileName = file.name.toLowerCase();
            const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
            
            if (!validTypes.includes(file.type) && !hasValidExtension) {
                alert('请选择支持的文件格式：图片(JPG/PNG/BMP/GIF)、PDF或Word文档！');
                return;
            }
            
            selectedFile = file;
            
            // 根据文件类型显示不同的预览
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    uploadArea.style.display = 'none';
                    previewContainer.style.display = 'block';
                    document.getElementById('resultContainer').style.display = 'none';
                };
                reader.readAsDataURL(file);
            } else {
                // PDF或Word文档，显示文件信息
                previewImage.style.display = 'none';
                uploadArea.style.display = 'none';
                previewContainer.style.display = 'block';
                document.getElementById('resultContainer').style.display = 'none';
                
                // 在预览区域显示文件信息
                const fileInfo = document.createElement('div');
                fileInfo.style.cssText = 'background: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; border: 2px solid #3f51b5;';
                fileInfo.innerHTML = `
                    <div style="font-size: 3em; margin-bottom: 10px;">${file.name.endsWith('.pdf') ? '📕' : '📘'}</div>
                    <div style="color: #303f9f; font-weight: bold; font-size: 1.2em;">${file.name}</div>
                    <div style="color: #666; margin-top: 5px;">大小: ${(file.size / 1024).toFixed(2)} KB</div>
                    <div style="color: #666; margin-top: 5px;">类型: ${file.name.endsWith('.pdf') ? 'PDF文档' : 'Word文档'}</div>
                `;
                
                // 插入到预览容器的开头
                const container = document.getElementById('previewContainer');
                const firstChild = container.firstChild;
                container.insertBefore(fileInfo, firstChild);
            }
        }
        
        function changeDocument() {
            uploadArea.style.display = 'block';
            previewContainer.style.display = 'none';
            document.getElementById('resultContainer').style.display = 'none';
            fileInput.value = '';
            selectedFile = null;
        }
        
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        async function askQuestion() {
            if (!selectedFile) return;
            
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('请输入问题！');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('question', question);
            
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('resultContainer');
            const askBtn = document.getElementById('askBtn');
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            askBtn.disabled = true;
            
            try {
                const response = await fetch('/ask', {
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
                askBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            const container = document.getElementById('resultContainer');
            const confidence = (data.score * 100).toFixed(1);
            
            let html = '<h3 style="color: #303f9f; margin-bottom: 20px; text-align: center;">✨ 分析结果</h3>';
            
            html += '<div class="answer-box">';
            html += `<div class="question-text">❓ ${data.question}</div>`;
            html += `<div class="answer-text">💡 ${data.answer}</div>`;
            html += `<div class="confidence">置信度: ${confidence}%</div>`;
            html += '<div class="confidence-bar">';
            html += `<div class="confidence-fill" style="width: ${confidence}%"></div>`;
            html += '</div>';
            html += '</div>';
            
            container.innerHTML = html;
            container.style.display = 'block';
        }
        
        document.getElementById('questionInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
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

@app.route('/ask', methods=['POST'])
def ask():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传文件'})
        
        file = request.files['image']
        question = request.form.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'error': '请输入问题'})
        
        filename = file.filename.lower()
        
        # 处理不同类型的文件
        if filename.endswith('.pdf'):
            # 处理PDF文件
            if not PDF_SUPPORT:
                return jsonify({'success': False, 'error': 'PDF支持未启用，请安装 pdf2image 和 poppler'})
            
            # 保存临时PDF文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                file.save(tmp_pdf.name)
                pdf_path = tmp_pdf.name
            
            try:
                # 转换PDF第一页为图片
                images = convert_from_path(pdf_path, first_page=1, last_page=1)
                if not images:
                    return jsonify({'success': False, 'error': 'PDF转换失败'})
                
                image = images[0].convert('RGB')
            finally:
                os.unlink(pdf_path)
                
        elif filename.endswith(('.doc', '.docx')):
            # 处理Word文档
            if not DOCX_READ_SUPPORT:
                return jsonify({'success': False, 'error': 'Word支持未启用，请安装 python-docx'})
            
            # 保存临时Word文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
                file.save(tmp_docx.name)
                docx_path = tmp_docx.name
            
            try:
                # 方法1: 如果有docx2pdf，转换为PDF再转图片
                if PDF_SUPPORT and DOCX_SUPPORT:
                    pdf_path = docx_path.replace('.docx', '.pdf')
                    try:
                        pythoncom.CoInitialize()
                        docx_to_pdf(docx_path, pdf_path)
                        images = convert_from_path(pdf_path, first_page=1, last_page=1)
                        image = images[0].convert('RGB')
                        os.unlink(pdf_path)
                    except Exception as e:
                        print(f"Word转换失败: {e}")
                        return jsonify({'success': False, 'error': f'Word文档转换失败: {str(e)}'})
                    finally:
                        pythoncom.CoUninitialize()
                else:
                    # 方法2: 提取文本（简化方案）
                    doc = Document(docx_path)
                    text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                    return jsonify({'success': False, 'error': 'Word文档需要转换为图片格式，请先将文档导出为PDF或截图'})
            finally:
                os.unlink(docx_path)
                
        else:
            # 处理图片文件
            image = Image.open(file.stream).convert('RGB')
        
        # 执行文档问答
        result = doc_qa(image=image, question=question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result[0]['answer'],
            'score': float(result[0]['score'])
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
    print("📄 启动文档解析师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:8001")
    print("🔍 文档解析师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:8001')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=8001, debug=False)
