#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
meta_*.jsonl -> clean_*_meta.jsonl

核心升级：
1. build_text() 里加入 price / categories 关键信息。
2. 对 details 做白名单筛选，去掉噪声字段。
3. 输出中额外保留 price, categories 字段，方便后续 prompt 设计使用。
"""

import argparse
import json
import os
from typing import Dict, Any, List, Optional


DETAIL_KEEP_KEYS = {
    "Brand",
    "Material",
    "Blade Material",
    "Item Dimensions LxWxH",
    "Style",
    "Color",
    "Size",
    "Product Dimensions",
}


def choose_image_url(images: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """
    从 images 列表中选择一个 image_url：
    1. 优先 hi_res
    2. 其次 large
    """
    if not images:
        return None

    for img in images:
        url = img.get("hi_res")
        if url:
            return url

    for img in images:
        url = img.get("large")
        if url:
            return url

    return None


def build_text(sample: Dict[str, Any]) -> str:
    """
    构造用于模型输入的文本：
    Title + Price + Categories + Features + Description + Filtered Details
    """
    parts: List[str] = []

    title = sample.get("title")
    if title:
        parts.append(title.strip())

    # Price（如果存在）
    price = sample.get("price")
    if price is not None:
        parts.append(f"Price: {price}")

    # Categories（可选）
    cats = sample.get("categories") or []
    cat_lines = [str(c).strip() for c in cats if str(c).strip()]
    if cat_lines:
        parts.append("[CATEGORIES]")
        parts.extend(cat_lines)

    # Features
    features = sample.get("features") or []
    feat_lines = [f"• {str(f).strip()}" for f in features if str(f).strip()]
    if feat_lines:
        parts.append("[FEATURES]")
        parts.extend(feat_lines)

    # Description
    desc = sample.get("description") or []
    desc_lines = [str(d).strip() for d in desc if str(d).strip()]
    if desc_lines:
        parts.append("[DESCRIPTION]")
        parts.extend(desc_lines)

    # Details（做白名单过滤）
    details = sample.get("details") or {}
    if isinstance(details, dict) and details:
        detail_lines = []
        for k, v in details.items():
            k = str(k).strip()
            v = str(v).strip()
            if not k or not v:
                continue
            if DETAIL_KEEP_KEYS and k not in DETAIL_KEEP_KEYS:
                continue
            detail_lines.append(f"{k}: {v}")
        if detail_lines:
            parts.append("[DETAILS]")
            parts.extend(detail_lines)

    text = "\n".join(parts).strip()
    return text


def process_file(
    input_path: str,
    output_path: str,
    min_rating_count: int = 10,
    category: str = "all_beauty",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    kept = 0
    skipped_no_label = 0
    skipped_few_ratings = 0
    skipped_no_image = 0
    skipped_empty_text = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            avg_rating = obj.get("average_rating")
            rating_num = obj.get("rating_number")

            # label 合法性
            if avg_rating is None or rating_num is None:
                skipped_no_label += 1
                continue

            try:
                avg_rating = float(avg_rating)
                rating_num = int(rating_num)
            except (TypeError, ValueError):
                skipped_no_label += 1
                continue

            if not (0 < avg_rating <= 5.0):
                skipped_no_label += 1
                continue

            if rating_num < min_rating_count:
                skipped_few_ratings += 1
                continue

            image_url = choose_image_url(obj.get("images") or [])
            if not image_url:
                skipped_no_image += 1
                continue

            text = build_text(obj)
            if not text:
                skipped_empty_text += 1
                continue

            parent_asin = obj.get("parent_asin") or obj.get("asin") or ""
            parent_asin = str(parent_asin).strip()
            if not parent_asin:
                parent_asin = f"no_asin_{total}"

            main_category = obj.get("main_category") or ""
            store = obj.get("store") or ""
            price = obj.get("price")
            categories = obj.get("categories") or []

            out_obj = {
                "id": f"{category}_{parent_asin}",
                "asin": parent_asin,
                "main_category": main_category,
                "store": store,
                "average_rating": avg_rating,
                "rating_number": rating_num,
                "price": price,
                "categories": categories,
                "image_url": image_url,
                "text": text,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input file: {input_path}")
    print(f"Total items         : {total}")
    print(f"Kept items          : {kept}")
    print(f"Skipped (no label)  : {skipped_no_label}")
    print(f"Skipped (few ratings < {min_rating_count}) : {skipped_few_ratings}")
    print(f"Skipped (no image)  : {skipped_no_image}")
    print(f"Skipped (empty text): {skipped_empty_text}")
    print(f"Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to meta_*.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output cleaned jsonl")
    parser.add_argument("--min_rating_count", type=int, default=10)
    parser.add_argument("--category", type=str, default="all_beauty")
    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        min_rating_count=args.min_rating_count,
        category=args.category,
    )


if __name__ == "__main__":
    main()
