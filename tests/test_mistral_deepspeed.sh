deepspeed  train/pipeline7b.py \
    --x_col "headline_no_ent_v2" \
    --exp_name "llama7B gpt_labels new_repo" \
    --y_col "pseudo_label" \
    --filename_headlines  "temp_pseudo_labels_v1.3.parquet" \
    --output_dir "results/model_news_cls_v2" \
    --lora_dim "8" \
    --bf16 \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --per_device_train_batch_size "4" \
    --per_device_eval_batch_size "4" \
    --pooler_type_logits "last" \
    --save_total_limit "1" \
    --learning_rate "0.00001" \
    --weight_decay "0.005" \
    --num_labels "3" \
    --gradient_accumulation_steps "5" \
    --evaluation_strategy "steps" \
    --logging_steps "25" \
    --save_steps "25" \
    --eval_steps "5" \
    --remove_unused_columns False \
    --is_debug \
    --deepspeed deepspeed_confs/ds_config_zero2_v1.json