export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main_train_osediff_residual.py \
    --pretrained_model_name_or_path=/data/pretrained/stable-diffusion-2-1-base \
    --ram_path=/data/pretrained/ram_swin_large_14m.pth \
    --val_path=/home/guojinpei/diff-car/testsets/LIVE1_color \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 1000 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --neg_prompt="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth" \
    --cfg_vsd=7.5 \
    --lora_rank=4 \
    --lambda_lpips=2 \
    --lambda_l2=1 \
    --lambda_vsd=1 \
    --lambda_vsd_lora=2 \
    --tracker_project_name "train_osediff_residual" \
    --no_vsd \
