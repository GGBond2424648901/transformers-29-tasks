#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的模型部署为 Web API 服务
需要安装: pip install flask
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import torch.nn.functional as F

# 创建 Flask 应用
app = Flask(__name__)
# 启用 CORS（允许跨域请求）
CORS(app)

# 全局变量存储模型
model = None
tokenizer = None
classifier = None

def load_model():
    """加载模型"""
    global model, tokenizer, classifier
    
    print("正在加载模型...")
    
    # 获取脚本所在目录的绝对路径
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "my_sentiment_model")
    
    print(f"模型路径: {model_path}")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件夹")
        print(f"   期望位置: {model_path}")
        print(f"   当前目录: {os.getcwd()}")
        print(f"\n💡 请确保在正确的目录运行脚本，或先训练模型")
        exit(1)
    
    # 方法1: 使用 pipeline（简单）
    classifier = pipeline("text-classification", model=model_path)
    
    # 方法2: 手动加载（更灵活）
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✓ 模型加载成功！")

@app.route('/')
def home():
    """首页"""
    return """
    <h1>情感分析 API</h1>
    <p>使用方法：</p>
    <ul>
        <li>POST /predict - 单个文本预测</li>
        <li>POST /batch_predict - 批量文本预测</li>
        <li>GET /health - 健康检查</li>
    </ul>
    <p>示例请求：</p>
    <pre>
    curl -X POST http://localhost:5000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"text": "这个产品很好"}'
    </pre>
    """

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """单个文本预测"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': '缺少 text 参数'}), 400
        
        text = data['text']
        
        # 方法1: 使用 pipeline
        if data.get('use_pipeline', True):
            result = classifier(text)[0]
            return jsonify({
                'text': text,
                'label': result['label'],
                'score': float(result['score']),
                'sentiment': '正面' if result['label'] == 'LABEL_1' else '负面'
            })
        
        # 方法2: 手动预测（返回更详细的信息）
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        negative_prob = probs[0][0].item()
        positive_prob = probs[0][1].item()
        
        return jsonify({
            'text': text,
            'sentiment': '正面' if positive_prob > negative_prob else '负面',
            'confidence': max(positive_prob, negative_prob),
            'probabilities': {
                'negative': negative_prob,
                'positive': positive_prob
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量文本预测"""
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({'error': '缺少 texts 参数'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts 必须是列表'}), 400
        
        # 批量预测
        results = classifier(texts)
        
        # 格式化结果
        formatted_results = []
        for text, result in zip(texts, results):
            formatted_results.append({
                'text': text,
                'label': result['label'],
                'score': float(result['score']),
                'sentiment': '正面' if result['label'] == 'LABEL_1' else '负面'
            })
        
        return jsonify({
            'count': len(texts),
            'results': formatted_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 启动前加载模型
    load_model()
    
    # 启动 Flask 服务
    print("\n" + "=" * 60)
    print("情感分析 API 服务已启动！")
    print("访问: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
