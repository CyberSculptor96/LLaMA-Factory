split=$1
if [ -z "$split" ]; then
  echo "Please provide a split (train/val/test)."
  exit 1
fi

python /wangbenyou/huanghj/workspace/project/amazon/LLaMA-Factory/user/construct_prompt.py \
    --split "$split" \
    --input /wangbenyou/huanghj/workspace/project/amazon/data/processed/splits_v2/${split}_all_beauty.jsonl \
    --output /wangbenyou/huanghj/workspace/project/amazon/LLaMA-Factory/data/prompts_v2/amazon_rating_${split}_v2.jsonl \
    --image_root_dir /wangbenyou/huanghj/workspace/project/amazon/data/cache/images/all_beauty \
    --no_short_variant
