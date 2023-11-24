python \
    train/pipeline7b.py \
    --x_col "headline_no_ent_v2" \
    --exp_name "mistral7B new_repo" \
    --y_col "RET_10D_pos" \
    --filename_headlines "temp_df_clusters_v3.parquet" \
    --output_dir "results/Llama_SFT" \
    --lora_dim "8" \
    --bf16 \
    --model_name  "mistralai/Mistral-7B-v0.1" \
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
    --push_to_hub True \
    --remove_unused_columns False \
