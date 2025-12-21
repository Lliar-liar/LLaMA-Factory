import torch
from PIL import Image
import cv2
import numpy as np
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
# 1. 基座模型路径 (自动下载或写本地路径)
BASE_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

# 2. 训练好的 LoRA 路径 (请修改为你最新的 checkpoint)
# 例如: "saves/qwen3vl-4b/lora/sft/checkpoint-500"
LORA_PATH = "/home/v-wangrui5/Qwen_ckpt/Qwen_abhuman/qwen3vl-7b/lora/sft/checkpoint-8000" 

# 3. 设备配置
DEVICE = "cuda"

# 4. 提示词 (必须与训练时保持一致)
# 训练时你的脚本用了随机 Prompt，这里建议用最通用的那个
PROMPT_TEXT = "Detect human anatomical anomalies in this image."
# ===========================================

def load_model():
    print(f"正在加载基座模型: {BASE_MODEL_PATH} ...")
    # 加载基座模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    
    print(f"正在加载 LoRA 权重: {LORA_PATH} ...")
    # 加载 LoRA
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    # 加载 Processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 设为评估模式
    model.eval()
    return model, processor

def parse_and_draw(image_path, model_response, output_path="result_output.jpg"):
    """
    解析模型输出的坐标并在图上画框
    """
    # 读取原图用于画图
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print("无法读取图片用于画图")
        return
    
    h_img, w_img = img_cv2.shape[:2]

    # 正则表达式提取: <|box_start|>(x1,y1),(x2,y2)<|box_end|><|object_ref_start|>label<|object_ref_end|>
    # Qwen 的 Grounding 输出格式比较标准
    pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|><\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    
    matches = re.findall(pattern, model_response)
    print(f"\n🔍 检测结果: 发现 {len(matches)} 个目标")

    found_anomaly = False
    
    for match in matches:
        x1_n, y1_n, x2_n, y2_n, label = match
        
        # 坐标反归一化 (0-1000 -> 实际像素)
        x1 = int(int(x1_n) / 1000 * w_img)
        y1 = int(int(y1_n) / 1000 * h_img)
        x2 = int(int(x2_n) / 1000 * w_img)
        y2 = int(int(y2_n) / 1000 * h_img)

        print(f"  - 目标: {label} | 坐标: [{x1}, {y1}, {x2}, {y2}]")

        # 颜色逻辑: 异常用红色，正常用绿色
        if "abnormal" in label.lower() or "anomaly" in label.lower():
            color = (0, 0, 255) # 红色 (BGR)
            found_anomaly = True
        elif "normal" in label.lower():
            color = (0, 255, 0) # 绿色
        else:
            color = (255, 0, 0) # 蓝色 (其他)

        # 画框 (线宽 2)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # 画标签背景
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(img_cv2, (x1, y1_label - label_size[1] - 10), (x1 + label_size[0], y1_label + baseline - 10), color, cv2.FILLED)
        
        # 写文字 (白色)
        cv2.putText(img_cv2, label, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存图片
    cv2.imwrite(output_path, img_cv2)
    print(f"🖼️  可视化结果已保存至: {output_path}")
    
    return found_anomaly

def run_inference(model, processor, image_path):
    print(f"\n正在分析图片: {image_path} ...")
    
    # 构造对话消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # 预处理
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

    # 推理生成
    # max_new_tokens 可以设大一点，防止框太多被截断
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # 解码输出 (去掉输入的 prompt 部分)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    # 打印原始输出 (调试用)
    # print(f"Raw Output: {output_text}")
    
    return output_text

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载模型 (只需加载一次)
    model, processor = load_model()
    
    # 2. 指定要测试的图片
    test_image = "/home/v-wangrui5/HumanRefiner/val/images/humangeneral9986.jpg"  # <--- 请替换为你的本地测试图片路径
    
    # 确保图片存在
    import os
    if not os.path.exists(test_image):
        print(f"错误: 找不到图片 {test_image}，请修改脚本中的路径。")
    else:
        # 3. 运行推理
        response = run_inference(model, processor, test_image)
        
        # 4. 解析并画图
        parse_and_draw(test_image, response, output_path="prediction_result.jpg")