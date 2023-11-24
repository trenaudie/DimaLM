torchrun \
    --standalone --nnodes=1 --nproc-per-node=2\
    train/pipeline7b.py \
    --x_col "headline_no_ent_v2" \
    --exp_name "llama7b gpt_labels" \
    --y_col "pseudo_label" \
    --filename_headlines  "temp_pseudo_labels_v1.3.parquet" \
    --output_dir "results/Mistral_Relevance" \
    --lora_dim "8" \
    --bf16 \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --num_train_epochs "2" \
    --per_device_train_batch_size "8" \
    --per_device_eval_batch_size "8" \
    --save_total_limit "1" \
    --learning_rate "0.00001" \
    --weight_decay "0.005" \
    --num_labels "3" \
    --gradient_accumulation_steps "5" \
    --evaluation_strategy "steps" \
    --logging_steps "25" \
    --save_steps "100" \
    --eval_steps "100" \
    --remove_unused_columns False \
    --use_mlp False \
    --push_to_hub True
