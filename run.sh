CUDA_VISIBLE_DEVICES=2 accelerate launch main_train_diff.py \
    --pretrained_model_name_or_path=/zfsauton2/home/jinpeig/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06 \
    --ram_path=/zfsauton2/home/jinpeig/.cache/huggingface/hub/ram_swin_large_14m.pth \
    --val_path=/zfsauton2/home/jinpeig/diff-car/testsets/LIVE1_color \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 500 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --neg_prompt="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth" \
    --cfg_vsd=7.5 \
    --lora_rank=256 \
    --lambda_lpips=2 \
    --lambda_l2=1 \
    --lambda_vsd=1 \
    --lambda_vsd_lora=1 \
    --tracker_project_name "train_diff_non_reg_256_lora_256" \

#    --dataset_txt_paths_list 'YOUR TXT FILE PATH','YOUR TXT FILE PATH'
#    --deg_file_path="params_realesrgan.yml"
#    --dataset_prob_paths_list 1,1 \
#    --train_batch_size=4 \