import cv2
import numpy as np
import re
import os
import subprocess
from tqdm import tqdm
from PIL import Image
import torch

# === vLLM & Qwen Utils ===
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置多进程启动方式，防止 vLLM 卡死
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ================= 配置区域 =================
# 1. 模型路径 (请修改为你 export 出来的 merged 模型路径)
MERGED_MODEL_PATH = "/home/v-wangrui5/Qwen_ckpt/Qwen_abhuman/qwen3vl-8b-v4-merged"

# 2. 显卡配置
# 如果显存不够 (OOM)，尝试降低到 0.8 或 0.7
GPU_MEMORY_UTILIZATION = 0.9 
MAX_MODEL_LEN = 8192         

# 3. 提示词 (保持与训练一致)
PROMPT_TEXT = "Please locate the abnormal and normal human parts in this image."

# 4. 视频配置
FRAME_INTERVAL = 3     # 正常采样间隔
ANOMALY_THRESHOLD = 0.1 # 判定不合格的阈值
USE_ADAPTIVE_SAMPLING = True # 开启自适应采样(发现异常后逐帧检测)
# ===========================================

def prepare_inputs_for_vllm(messages, processor):
    """
    Qwen3-VL 专用输入预处理，计算动态分辨率网格
    """
    # 1. 生成 Prompt 文本
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 处理视觉信息
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    # 如果未来处理视频输入，这里加 video

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs # 关键参数
    }

def init_vllm_model():
    print(f"Initializing Processor from: {MERGED_MODEL_PATH} ...")
    # 加载 Processor (用于处理图片 resize 和 prompt template)
    processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)

    print(f"Initializing vLLM Engine with Merged Model...")
    llm = LLM(
        model=MERGED_MODEL_PATH,
        enable_lora=False,          # <--- 关键：关闭 LoRA，因为权重已融合
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",           # A100 推荐使用 bf16
        limit_mm_per_prompt={"image": 1},
    )
    
    # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.1,  # 低温采样，保证检测稳定性
        top_p=0.8,
        max_tokens=512,   # 足够容纳多个框的坐标
        stop_token_ids=[151645, 151643] # Qwen EOS tokens
    )
    
    return llm, processor, sampling_params

def run_inference_vllm(llm, processor, sampling_params, pil_image):
    # 构造消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # 准备 vLLM 输入格式
    vllm_inputs = prepare_inputs_for_vllm(messages, processor)

    # 执行推理 (无需传入 lora_request)
    outputs = llm.generate(
        [vllm_inputs],
        sampling_params=sampling_params,
        use_tqdm=False
    )
    print(f"LLM Output: {outputs[0].outputs[0].text}")
    return outputs[0].outputs[0].text

def parse_and_draw_frame(frame_bgr, model_response):
    h_img, w_img = frame_bgr.shape[:2]
    
    # --- 1. 清洗数据：去除 <think> 内容 ---
    # 使用正则将 <think>...</think> 及其内部内容替换为空
    clean_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL).strip()
    
    # --- 2. 新的正则表达式 ---
    # 解释：
    # \((\d+),(\d+)\)  -> 匹配 (x1,y1)
    # ,                -> 匹配中间的逗号
    # \((\d+),(\d+)\)  -> 匹配 (x2,y2)
    # \s*              -> 匹配 0个或多个空格 (防止有的时候有空格)
    # ([^\n<]+)        -> 匹配标签内容 (直到换行符或下一个 < 符号出现)
    pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)\s*([^\n<]+)"
    
    matches = re.findall(pattern, clean_response)
    
    found_real_anomaly = False
    
    # 调试打印 (可选)
    # if matches:
    #     print(f"Frame detected: {matches}")
    
    for match in matches:
        # 正则提取出来的是字符串，转整数
        x1_n, y1_n, x2_n, y2_n, label = match
        label = label.strip() # 去除首尾空格
        
        # 反归一化坐标 (0-1000 -> 实际像素)
        x1 = int(int(x1_n) / 1000 * w_img)
        y1 = int(int(y1_n) / 1000 * h_img)
        x2 = int(int(x2_n) / 1000 * w_img)
        y2 = int(int(y2_n) / 1000 * h_img)

        # === 标签判定逻辑 ===
        label_lower = label.lower()
        
        # 1. 优先判定异常
        if "abnormal" in label_lower or "anomaly" in label_lower:
            color = (0, 0, 255) # Red (BGR)
            found_real_anomaly = True 
            
        # 2. 判定正常
        elif "normal" in label_lower:
            color = (0, 255, 0) # Green (BGR)
            
        # 3. 其他 (兜底)
        else:
            color = (255, 0, 0) # Blue

        # 画框
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        
        # 画文字标签
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(frame_bgr, (x1, y1_label - label_size[1] - 10), (x1 + label_size[0], y1_label + baseline - 10), color, cv2.FILLED)
        cv2.putText(frame_bgr, label, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame_bgr, found_real_anomaly

def merge_audio(video_no_audio, original_video_with_audio, output_path):
    print(f"Merging audio...")
    # 使用 ffmpeg 合并音轨
    command = [
        "ffmpeg", "-y", "-i", video_no_audio, "-i", original_video_with_audio,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except:
        return False

def process_video(video_path, llm, processor, sampling_params):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = f"{base_name}_silent_vllm.mp4"
    final_output_path = f"{base_name}_analyzed_vllm.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    print(f"Start processing video (Merged Model + vLLM): {video_path}")
    print(f"Strategy: {'Adaptive' if USE_ADAPTIVE_SAMPLING else 'Uniform'} Sampling")
    
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
            
            # BGR (OpenCV) -> RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # === 执行 vLLM 推理 ===
            try:
                response = run_inference_vllm(llm, processor, sampling_params, pil_img)
                # 解析并画框
                frame_annotated, has_real_anomaly = parse_and_draw_frame(frame, response)
            except Exception as e:
                print(f"Warning: Inference error at frame {frame_idx}: {e}")
                frame_annotated = frame
                has_real_anomaly = False
            
            # 自适应采样逻辑
            if has_real_anomaly:
                anomaly_frame_count += 1
                if USE_ADAPTIVE_SAMPLING:
                    next_inference_frame = frame_idx + 1 # 发现异常，下一帧继续测
                    msg = "ANOMALY! Checking next..."
                else:
                    next_inference_frame = frame_idx + FRAME_INTERVAL
                    msg = "ANOMALY DETECTED!"
                
                # 画面提示
                cv2.putText(frame_annotated, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 正常 (Normal或无框)，跳过间隔
                next_inference_frame = frame_idx + FRAME_INTERVAL
        else:
            frame_annotated = frame

        out_writer.write(frame_annotated)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out_writer.release()
    pbar.close()
    
    # 合并音频
    success = merge_audio(temp_video_path, video_path, final_output_path)
    if success and os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    elif not success:
        final_output_path = temp_video_path # 失败则保留无声版
    
    # 生成报告
    anomaly_ratio = anomaly_frame_count / sampled_count if sampled_count > 0 else 0
    is_video_abnormal = anomaly_ratio > ANOMALY_THRESHOLD
    
    print("\n" + "="*40)
    print(f"Final Output: {final_output_path}")
    print(f"Frames analyzed: {sampled_count} / {total_frames}")
    print(f"Frames with REAL anomalies: {anomaly_frame_count}")
    print(f"Anomaly Ratio: {anomaly_ratio:.2%}")
    if is_video_abnormal:
        print(f"🔴 结论: 不合格 (>{ANOMALY_THRESHOLD:.0%})")
    else:
        print(f"🟢 结论: 合格")
    print("="*40)

if __name__ == "__main__":
    # 1. 初始化引擎
    llm_engine, processor, params = init_vllm_model()
    
    # 2. 指定视频文件
    target_video = "/home/v-wangrui5/Bernoulli_s_Principle__Floating_Ball_.mp4" 
    
    # 3. 开始处理
    process_video(target_video, llm_engine, processor, params)