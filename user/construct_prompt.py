import json
import os
from typing import Dict, Any, List
from tqdm import tqdm

# 统一相对路径前缀，比如 all_beauty/B0157PBICG.jpg
IMAGE_REL_PREFIX = "/wangbenyou/huanghj/workspace/project/amazon/data/cache/images/all_beauty"


def build_prompt_full(item: Dict[str, Any]) -> str:
    """
    完整版 prompt：包含 full text（title + features + description + filtered details）。
    """
    category = item.get("main_category", "Unknown")
    store = item.get("store", "Unknown")
    text_desc = item.get("text", "")

    # 可选：简单把 categories 拼到 metadata 中
    cats = item.get("categories") or []
    cat_str = ", ".join([str(c).strip() for c in cats if str(c).strip()])
    if not cat_str:
        cat_str = "N/A"

    price = item.get("price", None)
    price_str = "N/A" if price is None else str(price)

    prompt = (
        "<image>\n"
        "You are an expert Amazon product analyst.\n\n"
        "Your task:\n"
        "Given the product image and the metadata below, estimate the **Average User Rating** this product receives on Amazon.\n\n"
        "### Product Metadata\n"
        f"- Main Category: {category}\n"
        f"- Store: {store}\n"
        f"- Price: {price_str}\n"
        f"- Categories: {cat_str}\n\n"
        "[PRODUCT DETAILS]\n"
        f"{text_desc}\n\n"
        "### Rating Scale\n"
        "- 1.0 = very bad quality or experience\n"
        "- 3.0 = average/acceptable\n"
        "- 5.0 = excellent quality and highly satisfying\n\n"
        "### Instructions\n"
        "- Use BOTH the product image and the metadata when judging.\n"
        "- Imagine many real users buying and using this product, then leaving honest ratings.\n\n"
        "### Output Format\n"
        "Respond with ONLY a single number between 1.0 and 5.0 (one decimal place), e.g. 3.5.\n"
        "Do NOT output any other text or explanation."
    )
    return prompt


def build_prompt_short(item: Dict[str, Any]) -> str:
    """
    短版 prompt：只用 title + 少量 summary 信息，
    强迫模型更多依赖图片（适合做 data augmentation，用于 train）。
    """
    category = item.get("main_category", "Unknown")
    store = item.get("store", "Unknown")
    text_desc = item.get("text", "")
    # 从 text 中截第一行作为 summary（一般是 title）
    first_line = text_desc.split("\n")[0] if text_desc else ""

    price = item.get("price", None)
    price_str = "N/A" if price is None else str(price)

    prompt = (
        "<image>\n"
        "You are an expert Amazon product analyst.\n\n"
        "Your task:\n"
        "Given the product image and the brief metadata below, estimate the **Average User Rating** this product receives on Amazon.\n\n"
        "### Product Metadata (Short)\n"
        f"- Main Category: {category}\n"
        f"- Store: {store}\n"
        f"- Price: {price_str}\n"
        f"- Product Summary: {first_line}\n\n"
        "### Rating Scale\n"
        "- 1.0 = very bad quality or experience\n"
        "- 3.0 = average/acceptable\n"
        "- 5.0 = excellent quality and highly satisfying\n\n"
        "### Instructions\n"
        "- Use BOTH the product image and the summary when judging.\n"
        "- Imagine many real users buying and using this product, then leaving honest ratings.\n\n"
        "### Output Format\n"
        "Respond with ONLY a single number between 1.0 and 5.0 (one decimal place), e.g. 3.5.\n"
        "Do NOT output any other text or explanation."
    )
    return prompt


def convert_split(
    split: str,
    input_file: str,
    image_root_dir: str,
    output_file: str,
    generate_short_variant_for_train: bool = True,
) -> None:
    """
    从 clean_*_meta.jsonl 转为 LLaMA-Factory 多模态 SFT 数据（jsonl）。

    :param split: "train" / "val" / "test"
    :param input_file: clean_*_meta.jsonl
    :param image_root_dir: 本地图片根目录，例如 ".../data/cache/images/all_beauty"
    :param output_file: 输出 jsonl
    :param generate_short_variant_for_train: train split 是否为每条样本生成 short prompt 变体
    """
    is_train = (split == "train")

    total_lines = 0
    written = 0
    skipped_no_image = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Converting {split}"):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = item.get("asin")
            rating = item.get("average_rating")
            if not asin or rating is None:
                continue

            # 本地图片路径（假设已经下载为 {asin}.jpg）
            img_filename = f"{asin}.jpg"
            img_abs_path = os.path.join(image_root_dir, img_filename)
            if not os.path.exists(img_abs_path):
                skipped_no_image += 1
                continue

            # 写相对路径，便于 LLaMA-Factory 配置 image_folder
            img_rel_path = f"{IMAGE_REL_PREFIX}/{img_filename}"

            # label：一位小数
            rating_str = f"{float(rating):.1f}"

            # 1) full prompt
            full_prompt = build_prompt_full(item)
            entry_full = {
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": rating_str,
                    },
                ],
                "images": [img_rel_path],
            }
            fout.write(json.dumps(entry_full, ensure_ascii=False) + "\n")
            written += 1

            # 2) 只在 train 上加一个 short variant
            if is_train and generate_short_variant_for_train:
                short_prompt = build_prompt_short(item)
                entry_short = {
                    "messages": [
                        {
                            "role": "user",
                            "content": short_prompt,
                        },
                        {
                            "role": "assistant",
                            "content": rating_str,
                        },
                    ],
                    "images": [img_rel_path],
                }
                fout.write(json.dumps(entry_short, ensure_ascii=False) + "\n")
                written += 1

    print(f"\n[{split}] total input lines: {total_lines}")
    print(f"[{split}] written samples   : {written}")
    print(f"[{split}] skipped(no image) : {skipped_no_image}")
    print(f"Output file: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "val", "test"])
    parser.add_argument("--input", type=str, required=True,
                        help="clean_*_meta.jsonl for this split")
    parser.add_argument("--image_root_dir", type=str, required=True,
                        help="absolute path to local image dir, e.g. .../images/all_beauty")
    parser.add_argument("--output", type=str, required=True,
                        help="output jsonl for LLaMA-Factory")
    parser.add_argument("--no_short_variant", action="store_true",
                        help="disable short prompt variant for train split")
    args = parser.parse_args()

    convert_split(
        split=args.split,
        input_file=args.input,
        image_root_dir=args.image_root_dir,
        output_file=args.output,
        generate_short_variant_for_train=not args.no_short_variant,
    )
