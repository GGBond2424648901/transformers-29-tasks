#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态估计实时检测 - 支持摄像头和视频文件 🎥
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import cv2
import numpy as np
from PIL import Image
import time

# 保存原始的cv2.imshow，避免被ultralytics patch
_original_imshow = cv2.imshow

# 使用YOLO进行人体检测
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
    USE_YOLO = True
    print("✅ YOLO模型加载成功")
    # 恢复原始的imshow函数
    cv2.imshow = _original_imshow
except Exception as e:
    USE_YOLO = False
    yolo_model = None
    print(f"⚠️ YOLO加载失败: {e}")

# 使用OpenPose进行姿态估计
try:
    from controlnet_aux import OpenposeDetector
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    USE_OPENPOSE = True
    print("✅ OpenPose模型加载成功")
except Exception as e:
    USE_OPENPOSE = False
    pose_detector = None
    print(f"⚠️ OpenPose加载失败: {e}")

print("\n" + "=" * 70)
print("🎥 姿态估计实时检测系统")
print("=" * 70)
print("\n选择检测模式：")
print("1. 摄像头实时检测")
print("2. 视频文件检测")
print("3. 退出")

def detect_pose_in_frame(frame):
    """对单帧图像进行姿态检测"""
    # 转换为PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # YOLO检测人数
    num_people = 0
    if USE_YOLO and yolo_model:
        try:
            results = yolo_model(pil_image, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # person
                        num_people += 1
        except:
            pass
    
    # OpenPose检测骨骼
    pose_image = None
    if USE_OPENPOSE and pose_detector:
        try:
            pose_image = pose_detector(pil_image, detect_resolution=384, image_resolution=384)
            # 转换回OpenCV格式
            pose_array = np.array(pose_image)
            pose_bgr = cv2.cvtColor(pose_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"姿态检测失败: {e}")
            pose_bgr = frame.copy()
    else:
        pose_bgr = frame.copy()
    
    return pose_bgr, num_people

def camera_detection():
    """摄像头实时检测"""
    print("\n🎥 启动摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 摄像头已启动")
    print("\n操作说明：")
    print("  - 按 'q' 退出")
    print("  - 按 's' 截图保存")
    print("  - 按 'p' 暂停/继续")
    
    paused = False
    frame_count = 0
    fps_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            # 检测姿态
            pose_frame, num_people = detect_pose_in_frame(frame)
            
            # 计算FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            else:
                fps = 0
            
            # 添加信息文字
            if fps > 0:
                cv2.putText(pose_frame, f'FPS: {fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(pose_frame, f'People: {num_people}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(pose_frame, 'Press Q to quit', (10, pose_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('Pose Detection - Camera', pose_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'pose_capture_{int(time.time())}.jpg'
            cv2.imwrite(filename, pose_frame)
            print(f"📸 截图已保存: {filename}")
        elif key == ord('p'):
            paused = not paused
            print("⏸️ 暂停" if paused else "▶️ 继续")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 摄像头检测结束")

def video_detection():
    """视频文件检测"""
    print("\n📹 请输入视频文件路径：")
    video_path = input("> ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n✅ 视频信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧率: {fps} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {total_frames/fps:.1f} 秒")
    
    print("\n是否保存处理后的视频？(y/n)")
    save_video = input("> ").strip().lower() == 'y'
    
    out = None
    if save_video:
        output_path = f'pose_output_{int(time.time())}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"📹 将保存到: {output_path}")
    
    print("\n操作说明：")
    print("  - 按 'q' 退出")
    print("  - 按 'p' 暂停/继续")
    print("  - 按 '→' 快进10帧")
    print("  - 按 '←' 后退10帧")
    
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ 视频处理完成")
                break
            
            frame_idx += 1
            
            # 检测姿态
            pose_frame, num_people = detect_pose_in_frame(frame)
            
            # 添加信息文字
            progress = (frame_idx / total_frames) * 100
            cv2.putText(pose_frame, f'Frame: {frame_idx}/{total_frames} ({progress:.1f}%)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(pose_frame, f'People: {num_people}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存视频
            if out is not None:
                out.write(pose_frame)
            
            # 显示画面
            cv2.imshow('Pose Detection - Video', pose_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️ 暂停" if paused else "▶️ 继续")
        elif key == 83:  # 右箭头
            frame_idx = min(frame_idx + 10, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"⏩ 快进到第 {frame_idx} 帧")
        elif key == 81:  # 左箭头
            frame_idx = max(frame_idx - 10, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"⏪ 后退到第 {frame_idx} 帧")
    
    cap.release()
    if out is not None:
        out.release()
        print(f"\n✅ 视频已保存: {output_path}")
    cv2.destroyAllWindows()

def main():
    """主函数"""
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == '1':
            camera_detection()
        elif choice == '2':
            video_detection()
        elif choice == '3':
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选择，请输入 1、2 或 3")

if __name__ == '__main__':
    if not USE_YOLO and not USE_OPENPOSE:
        print("\n❌ 错误：YOLO和OpenPose都未加载成功")
        print("请确保已安装：")
        print("  pip install ultralytics")
        print("  pip install controlnet-aux")
    else:
        main()
