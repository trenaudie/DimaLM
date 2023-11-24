python train/pipeline7b.py \
    --x_col "headline" \
    --exp_name "llama7B no_contexts new_repo" \
    --y_col "RET_10D_pos" \
    --filename_headlines  "news_headlines_v3.2.parquet" \
    --output_dir "results/model_news_cls_v2" \
    --lora_dim "8" \
    --bf16 \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --per_device_train_batch_size "4" \
    --per_device_eval_batch_size "4" \
    --save_total_limit "1" \
    --learning_rate "0.00001" \
    --weight_decay "0.005" \
    --num_labels "2" \
    --gradient_accumulation_steps "5" \
    --evaluation_strategy "steps" \
    --logging_steps "25" \
    --save_steps "25" \
    --eval_steps "25" \
    --remove_unused_columns False \
    --add_context False \
    --is_debug \
