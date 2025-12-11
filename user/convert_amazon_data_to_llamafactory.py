import json
import os
from tqdm import tqdm # 如果没有安装，请运行 pip install tqdm

# ==================== 配置区域 ====================
# 1. 你的原始 JSONL 数据路径
split = 'test'  # 'train' 或 'val'
INPUT_FILE = f"./data/processed/splits/{split}_all_beauty.jsonl" 

# 2. 图片存储的根目录 (请务必使用绝对路径，避免后续训练找不到文件)
# 例如: "/home/user/projects/amazon_project/images_cache"
IMAGE_ROOT_DIR = "/wangbenyou/huanghj/workspace/project/amazon/data/cache/images/all_beauty" 

# 3. 输出给 LLaMA-Factory 的文件路径
OUTPUT_FILE = f"amazon_rating_{split}.json"
# ================================================

def format_user_prompt(item):
    """
    根据 metadata 构建输入给模型的 Prompt。
    这里我们将分类、店铺和商品详情都整合进去，辅助模型判断。
    """
    category = item.get("main_category", "Unknown")
    store = item.get("store", "Unknown")
    text_desc = item.get("text", "")
    
    # 构造 Prompt 模板
    # 注意：<image> 标签由 LLaMA-Factory 处理，这里我们在文本中通过 prompt 暗示图片的存在
    prompt = (
        f"<image>\n" # Qwen2-VL 建议图片放在最前
        f"You are an expert product analyst analyzing an Amazon product listing.\n"
        f"Based on the product image and the metadata provided below, predict the Average User Rating (from 1.0 to 5.0).\n\n"
        f"### Product Metadata:\n"
        f"- Main Category: {category}\n"
        f"- Store: {store}\n"
        f"- Product Details: {text_desc}\n\n"
        f"### Task:\n"
        f"Predict the average rating score directly."
    )
    return prompt

def main():
    print(f"开始处理数据: {INPUT_FILE} ...")
    
    transformed_data = []
    skipped_count = 0
    
    # 检查图片目录是否存在
    if not os.path.exists(IMAGE_ROOT_DIR):
        print(f"错误: 图片目录不存在 -> {IMAGE_ROOT_DIR}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 读取所有行
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Converting"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            asin = item.get("asin")
            rating = item.get("average_rating")
            
            # 1. 严格的数据校验
            if not asin or rating is None:
                continue
                
            # 2. 构建图片绝对路径 (匹配你的逻辑: <asin>.jpg)
            image_filename = f"{asin}.jpg"
            image_abs_path = os.path.join(IMAGE_ROOT_DIR, image_filename)
            
            # 3. 关键：验证图片是否存在于本地
            # 如果图片下载失败了，这条数据必须扔掉，否则训练时会中断
            if not os.path.exists(image_abs_path):
                skipped_count += 1
                continue
            
            # 4. 构造 LLaMA-Factory 数据单元
            entry = {
                "messages": [
                    {
                        "content": format_user_prompt(item),
                        "role": "user"
                    },
                    {
                        # 必须转为字符串，建议保留一位小数，统一格式
                        "content": f"{float(rating):.1f}", 
                        "role": "assistant"
                    }
                ],
                "images": [
                    image_abs_path # 必须是列表
                ]
            }
            
            transformed_data.append(entry)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    print(f"\n================ 处理完成 ================")
    print(f"总输入行数: {len(lines)}")
    print(f"有效转换: {len(transformed_data)}")
    print(f"跳过(图片缺失/数据不全): {skipped_count}")
    print(f"输出文件: {os.path.abspath(OUTPUT_FILE)}")
    print(f"请将该文件路径注册到 data/dataset_info.json 中")

if __name__ == "__main__":
    main()