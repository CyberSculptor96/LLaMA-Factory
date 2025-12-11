import json
import torch
import re
import os
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info  # 确保你安装了 qwen-vl-utils

# ================= 配置区域 =================
# 1. 模型路径
BASE_MODEL_PATH = "/sds_wangby/models/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "saves/qwen2_5vl-7b/lora/sft/checkpoint-1250" # 替换为你表现最好的 checkpoint

# 2. 测试数据路径 (请确保这是验证集或测试集)
TEST_DATA_PATH = "data/amazon_rating_test.json"  # 如果没有拆分，暂时用原文件，但最好是没见过的数据
# 如果是绝对路径，请修改 IMAGE_ROOT 为 ""
IMAGE_ROOT = "" 

# 3. 输出结果路径
OUTPUT_RESULT_PATH = "evaluation_results.json"

# 4. 推理参数
MAX_PIXELS = 1003520 # 必须和训练时保持一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def extract_rating(text):
    """
    从模型输出中提取数字评分。
    支持格式: "3.5", "Rating: 3.5", "It is 3.5" 等
    """
    # 匹配浮点数或整数
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        # 取最后一个数字通常最保险 (例如: "Predicted rating is 3.5")
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def main():
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    except Exception as e:
        print(f"Warning: Could not load LoRA adapter. Error: {e}")
        print("Running with Base Model only!")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, min_pixels=256, max_pixels=MAX_PIXELS)
    
    print(f"Loading test data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 结果容器
    predictions = []
    ground_truths = []
    results_log = []

    print("Starting inference...")
    model.eval()
    
    # 逐条推理 (Batch size = 1 为最稳妥方式，避免不同分辨率图片的 Padding 问题)
    for i, item in tqdm(enumerate(test_data), total=len(test_data)):
        # 1. 解析数据
        # LLaMA-Factory 格式: messages[0] 是 user, messages[1] 是 assistant (GT)
        user_msg = item["messages"][0]
        gt_msg = item["messages"][1]
        
        # 获取 GT 分数
        try:
            gt_score = float(gt_msg["content"])
        except ValueError:
            print(f"Skipping sample {i}: GT is not a number ({gt_msg['content']})")
            continue

        # 获取图片路径
        image_path = item["images"][0]
        if IMAGE_ROOT:
            image_path = os.path.join(IMAGE_ROOT, image_path)

        # 2. 构造 Prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_msg["content"].replace("<image>", "")}, # 去掉 text 里的占位符，通过 type: image 传入
                ],
            }
        ]

        # 3. 预处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        # 4. 生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, temperature=0.01) # 低温采样保证确定性
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 5. 提取预测分
        pred_score = extract_rating(output_text)
        
        # 记录
        if pred_score is not None:
            predictions.append(pred_score)
            ground_truths.append(gt_score)
            
            results_log.append({
                "image": image_path,
                "gt": gt_score,
                "pred": pred_score,
                "raw_output": output_text,
                "diff": abs(pred_score - gt_score)
            })
        else:
            print(f"Sample {i} failed parsing. Output: {output_text}")

    # ================= 计算指标 =================
    if not predictions:
        print("No valid predictions found.")
        return

    preds_np = np.array(predictions)
    gts_np = np.array(ground_truths)

    mae = np.mean(np.abs(preds_np - gts_np))
    mse = np.mean((preds_np - gts_np) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算 Accuracy (Tolerance = 0.5)
    # 如果预测值和真实值差距小于 0.5，视为“准确”
    acc_05 = np.mean(np.abs(preds_np - gts_np) <= 0.5)

    print("\n" + "="*30)
    print("Evaluation Report")
    print("="*30)
    print(f"Total Samples: {len(preds_np)}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"MSE (均方误差):    {mse:.4f}")
    print(f"RMSE (均方根误差):  {rmse:.4f}")
    print(f"Acc@0.5 (容差0.5):  {acc_05:.2%}")
    print("="*30)

    # 保存结果
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "acc_05": acc_05},
            "details": results_log
        }, f, indent=2)
    print(f"Detailed results saved to {OUTPUT_RESULT_PATH}")

if __name__ == "__main__":
    main()