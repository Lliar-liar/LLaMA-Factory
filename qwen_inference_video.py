import torch
from PIL import Image
import cv2
import numpy as np
import re
import os
import subprocess
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
# 1. 模型路径
BASE_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
LORA_PATH = "/home/v-wangrui5/Qwen_ckpt/Qwen_abhuman/qwen3vl-7b/lora/sft/checkpoint-8000" 

# 2. 设备
DEVICE = "cuda"

# 3. 提示词
PROMPT_TEXT = "Detect human anatomical anomalies in this image."

# 4. 视频处理配置
FRAME_INTERVAL = 4      # 采样间隔 (每隔多少帧检测一次)
ANOMALY_THRESHOLD = 0.1 # 异常率阈值

# 5. 采样策略开关 (新增)
USE_ADAPTIVE_SAMPLING = False # True: 自适应采样 (发现异常后逐帧检测) | False: 均匀采样 (始终固定间隔)
# ===========================================

def load_model():
    print(f"Loading Base Model: {BASE_MODEL_PATH} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA Adapter: {LORA_PATH} ...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, processor

def run_inference_on_image(model, processor, pil_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def parse_and_draw_frame(frame_bgr, model_response):
    h_img, w_img = frame_bgr.shape[:2]
    pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|><\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    matches = re.findall(pattern, model_response)
    
    found_anomaly = False
    anomaly_details = []

    for match in matches:
        x1_n, y1_n, x2_n, y2_n, label = match
        
        x1 = int(int(x1_n) / 1000 * w_img)
        y1 = int(int(y1_n) / 1000 * h_img)
        x2 = int(int(x2_n) / 1000 * w_img)
        y2 = int(int(y2_n) / 1000 * h_img)

        is_abnormal = "abnormal" in label.lower() or "anomaly" in label.lower()
        
        if is_abnormal:
            color = (0, 0, 255) # Red
            found_anomaly = True
            anomaly_details.append(label)
        elif "normal" in label.lower():
            color = (0, 255, 0) # Green
        else:
            color = (255, 0, 0) # Blue

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        
        label_str = label
        label_size, baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(frame_bgr, (x1, y1_label - label_size[1] - 10), (x1 + label_size[0], y1_label + baseline - 10), color, cv2.FILLED)
        cv2.putText(frame_bgr, label_str, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame_bgr, found_anomaly, anomaly_details

def merge_audio(video_no_audio, original_video_with_audio, output_path):
    print(f"Merging audio from {original_video_with_audio} to {video_no_audio}...")
    command = [
        "ffmpeg", "-y",
        "-i", video_no_audio,
        "-i", original_video_with_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✅ Audio merge successful! Final video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg merge failed. Error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found.")
        return False

def process_video(video_path, model, processor):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = f"{base_name}_silent_temp.mp4"
    final_output_path = f"{base_name}_analyzed.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    print(f"Start processing video: {video_path}")
    print(f"Total frames: {total_frames}, Resolution: {width}x{height}, FPS: {fps}")
    
    # === 打印当前的采样策略 ===
    if USE_ADAPTIVE_SAMPLING:
        print(f"Strategy: [ADAPTIVE] Check every {FRAME_INTERVAL} frames, switch to continuous check on anomaly.")
    else:
        print(f"Strategy: [UNIFORM] Strict sampling every {FRAME_INTERVAL} frames.")

    frame_idx = 0
    sampled_count = 0
    anomaly_frame_count = 0
    next_inference_frame = 0 
    
    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 判断当前帧是否需要推理
        if frame_idx == next_inference_frame:
            sampled_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            response = run_inference_on_image(model, processor, pil_img)
            frame_annotated, has_anomaly, details = parse_and_draw_frame(frame, response)
            
            if has_anomaly:
                anomaly_frame_count += 1
                
                # === 核心修改逻辑 ===
                if USE_ADAPTIVE_SAMPLING:
                    # 自适应模式：下一帧立即检测
                    next_inference_frame = frame_idx + 1
                    msg = "ANOMALY! Checking next frame..."
                else:
                    # 均匀模式：依然跳过间隔
                    next_inference_frame = frame_idx + FRAME_INTERVAL
                    msg = "ANOMALY DETECTED!"

                # 提示文字
                cv2.putText(frame_annotated, msg, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 无异常，跳过间隔
                next_inference_frame = frame_idx + FRAME_INTERVAL
        else:
            frame_annotated = frame

        out_writer.write(frame_annotated)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out_writer.release()
    pbar.close()
    
    print("Processing visualization finished. Merging audio...")
    success = merge_audio(temp_video_path, video_path, final_output_path)
    if success and os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    elif not success:
        final_output_path = temp_video_path

    # === 最终报告 ===
    anomaly_ratio = anomaly_frame_count / sampled_count if sampled_count > 0 else 0
    is_video_abnormal = anomaly_ratio > ANOMALY_THRESHOLD
    
    print("\n" + "="*40)
    print(f"Processing Complete.")
    print(f"Final Output saved to: {final_output_path}")
    print(f"Frames analyzed ({'Adaptive' if USE_ADAPTIVE_SAMPLING else 'Uniform'}): {sampled_count} / {total_frames}")
    print(f"Frames with anomalies: {anomaly_frame_count}")
    print(f"Anomaly Ratio: {anomaly_ratio:.2%}")
    print("-" * 40)
    
    if is_video_abnormal:
        print(f"🔴 结论: 视频包含异常人体结构 (不合格)")
        print(f"   判定阈值: > {ANOMALY_THRESHOLD:.0%}")
    else:
        print(f"🟢 结论: 视频人体结构正常 (合格)")
    print("="*40)

if __name__ == "__main__":
    # 1. 加载模型
    model, processor = load_model()
    
    # 2. 视频路径
    target_video = "/home/v-wangrui5/guitar_Cmajor_chord_2.mp4" 
    
    # 3. 运行
    process_video(target_video, model, processor)