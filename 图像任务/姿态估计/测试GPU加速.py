#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速性能测试
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
import time
import cv2
import numpy as np
from PIL import Image

print("=" * 70)
print("🚀 GPU加速性能测试")
print("=" * 70)

# 1. 检查PyTorch CUDA支持
print("\n【1】PyTorch CUDA检测")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA不可用！请安装CUDA版本的PyTorch")
    print("   安装命令: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# 2. 测试YOLO GPU加速
print("\n【2】YOLO GPU加速测试")
try:
    from ultralytics import YOLO
    
    # 加载模型
    print("加载YOLO模型...")
    model = YOLO('yolov8n.pt')
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_image)
    
    # CPU测试
    print("\n测试CPU模式...")
    model.to('cpu')
    start = time.time()
    for i in range(10):
        results = model(test_pil, verbose=False, device='cpu')
    cpu_time = (time.time() - start) / 10
    print(f"CPU平均耗时: {cpu_time*1000:.2f} ms/帧")
    print(f"CPU FPS: {1/cpu_time:.2f}")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n测试GPU模式...")
        model.to('cuda')
        
        # 预热GPU
        for i in range(3):
            results = model(test_pil, verbose=False, device='cuda')
        
        start = time.time()
        for i in range(10):
            results = model(test_pil, verbose=False, device='cuda')
        gpu_time = (time.time() - start) / 10
        print(f"GPU平均耗时: {gpu_time*1000:.2f} ms/帧")
        print(f"GPU FPS: {1/gpu_time:.2f}")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        if gpu_time >= cpu_time:
            print("⚠️ 警告: GPU速度没有比CPU快，可能存在问题！")
    
    print("✅ YOLO测试完成")
    
except Exception as e:
    print(f"❌ YOLO测试失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试OpenPose GPU加速
print("\n【3】OpenPose GPU加速测试")
try:
    from controlnet_aux import OpenposeDetector
    
    print("加载OpenPose模型...")
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    # 检查模型是否有GPU支持
    if hasattr(detector, 'model_pose'):
        print(f"OpenPose模型类型: {type(detector.model_pose)}")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        # CPU测试
        print("\n测试CPU模式...")
        if hasattr(detector.model_pose, 'to'):
            detector.model_pose = detector.model_pose.to('cpu')
        
        start = time.time()
        for i in range(5):
            result = detector(test_pil, detect_resolution=192, image_resolution=192, hand_and_face=False)
        cpu_time = (time.time() - start) / 5
        print(f"CPU平均耗时: {cpu_time*1000:.2f} ms/帧")
        print(f"CPU FPS: {1/cpu_time:.2f}")
        
        # GPU测试
        if torch.cuda.is_available():
            print("\n测试GPU模式...")
            if hasattr(detector.model_pose, 'to'):
                detector.model_pose = detector.model_pose.to('cuda')
                
                # 预热GPU
                for i in range(2):
                    result = detector(test_pil, detect_resolution=192, image_resolution=192, hand_and_face=False)
                
                start = time.time()
                for i in range(5):
                    result = detector(test_pil, detect_resolution=192, image_resolution=192, hand_and_face=False)
                gpu_time = (time.time() - start) / 5
                print(f"GPU平均耗时: {gpu_time*1000:.2f} ms/帧")
                print(f"GPU FPS: {1/gpu_time:.2f}")
                print(f"加速比: {cpu_time/gpu_time:.2f}x")
                
                if gpu_time >= cpu_time:
                    print("⚠️ 警告: GPU速度没有比CPU快，OpenPose可能未使用GPU！")
            else:
                print("⚠️ OpenPose模型不支持.to()方法，可能无法使用GPU")
    else:
        print("⚠️ 无法访问OpenPose内部模型")
    
    print("✅ OpenPose测试完成")
    
except Exception as e:
    print(f"❌ OpenPose测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 综合性能评估
print("\n【4】综合性能评估")
print("=" * 70)

if torch.cuda.is_available():
    print("✅ GPU加速可用")
    print("\n推荐配置:")
    print("  - detect_resolution: 192 (平衡速度和精度)")
    print("  - hand_and_face: False (提高速度)")
    print("  - JPEG质量: 70 (减少传输时间)")
    print("  - 跳帧: 1 (不跳帧，实时性最好)")
    print("\n预期性能:")
    print("  - YOLO: 5-10ms/帧")
    print("  - OpenPose: 30-50ms/帧")
    print("  - 总延迟: 40-70ms/帧")
    print("  - FPS: 15-25")
else:
    print("❌ GPU不可用，使用CPU模式")
    print("\n预期性能:")
    print("  - YOLO: 50-100ms/帧")
    print("  - OpenPose: 200-400ms/帧")
    print("  - 总延迟: 300-500ms/帧")
    print("  - FPS: 2-3")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
