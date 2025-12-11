#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 clean_meta.jsonl 做 asin 级别的 train/val/test 划分
确保同一个 asin 不会泄露到不同 split
"""

import json
import random
import argparse
from collections import defaultdict


def split_by_asin(clean_file, out_train, out_val, out_test,
                  train_ratio=0.8, val_ratio=0.1, seed=42):

    random.seed(seed)

    asin_to_items = defaultdict(list)

    # 1) 按 asin 收集样本
    with open(clean_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj["asin"]
            asin_to_items[asin].append(obj)

    asins = list(asin_to_items.keys())
    random.shuffle(asins)

    N = len(asins)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_asins = set(asins[:n_train])
    val_asins = set(asins[n_train:n_train+n_val])
    test_asins = set(asins[n_train+n_val:])

    print(f"Total ASINs: {N}")
    print(f"Train ASINs: {len(train_asins)}")
    print(f"Val ASINs:   {len(val_asins)}")
    print(f"Test ASINs:  {len(test_asins)}")

    # 2) 写出三个文件
    fout_train = open(out_train, "w", encoding="utf-8")
    fout_val = open(out_val, "w", encoding="utf-8")
    fout_test = open(out_test, "w", encoding="utf-8")

    for asin, items in asin_to_items.items():
        if asin in train_asins:
            fout = fout_train
        elif asin in val_asins:
            fout = fout_val
        else:
            fout = fout_test

        for obj in items:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout_train.close()
    fout_val.close()
    fout_test.close()

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_file", type=str, required=True,
                        help="输入的 clean_meta.jsonl 文件路径")
    parser.add_argument("--output_train", type=str, required=True,
                        help="输出的 train split 文件路径")
    parser.add_argument("--output_val", type=str, required=True,
                        help="输出的 val split 文件路径")
    parser.add_argument("--output_test", type=str, required=True,
                        help="输出的 test split 文件路径")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    split_by_asin(
        clean_file=args.clean_file,
        out_train=args.output_train,
        out_val=args.output_val,
        out_test=args.output_test,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
