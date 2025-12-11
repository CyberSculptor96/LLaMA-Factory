base=/wangbenyou/huanghj/workspace/project/amazon/data/processed

python /wangbenyou/huanghj/workspace/project/amazon/LLaMA-Factory/user/split_metadata.py \
  --clean_file ${base}/clean_all_beauty_meta_pro.jsonl \
  --output_train ${base}/clean_train.jsonl \
  --output_val ${base}/clean_val.jsonl \
  --output_test ${base}/clean_test.jsonl