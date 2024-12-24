export CUDA_VISIBLE_DEVICES=4
python debug.py \
-i testsets/LIVE1_color \
-o debug \
--osediff_path /home/guojinpei/diff-car/training_results/train_diff_512_large_dataset_no_vsd/2024-12-13_22-01-24/checkpoints/model_max_train_steps.pkl \
--pretrained_model_name_or_path /data/pretrained/stable-diffusion-2-1-base \
--ram_ft_path /data/pretrained/DAPE.pth \
--ram_path /data/pretrained/ram_swin_large_14m.pth