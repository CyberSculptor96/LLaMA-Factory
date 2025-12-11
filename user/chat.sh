CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /sds_wangby/models/Qwen2.5-VL-7B-Instruct \
    --adapter_name_or_path saves/qwen2_5vl-7b/lora/sft/checkpoint-250 \
    --template qwen2_vl \
    --image_max_pixels 1003520