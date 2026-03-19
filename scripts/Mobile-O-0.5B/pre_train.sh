RUN_NAME=Mobile-O-0.5B-Pretrain
OUTPUT_FOLDER=checkpoints
if [ ! -d "$OUTPUT_FOLDER/llava-fastvithd_0.5b_stage3" ]; then
    mkdir -p $OUTPUT_FOLDER
    wget -nc https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip -O $OUTPUT_FOLDER/llava-fastvithd_0.5b_stage3.zip
    unzip -o $OUTPUT_FOLDER/llava-fastvithd_0.5b_stage3.zip -d $OUTPUT_FOLDER/  
    rm $OUTPUT_FOLDER/llava-fastvithd_0.5b_stage3.zip
fi
torchrun --nproc_per_node=1 mobileo/train/train.py \
    --deepspeed ./deepspeed_scripts/zero3.json \
    --diffusion_name_or_path Efficient-Large-Model/Sana_600M_512px_diffusers \
    --vlm_num_layers 4 \
    --model_name_or_path $OUTPUT_FOLDER/llava-fastvithd_0.5b_stage3 \
    --version qwen \
    --data_type mix \
    --image_folder data/Mobile-O-Pre-Train \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --fp16 False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER/$RUN_NAME \
    --aspect_ratio_size 512 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 2e-6}'
    --max_grad_norm 0.5 \
    --adam_beta2 0.95 \
    --model_max_length 512 \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME.txt
