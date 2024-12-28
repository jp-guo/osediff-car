export CUDA_VISIBLE_DEVICES=4
python main_test_osediff_residual.py \
-i /data/dataset/CAR/Urban100 \
-o test_results/Urban100_OSEDIFF_residual \
--osediff_path /home/guojinpei/diff-car/training_results/train_osediff_residual/2024-12-26_09-13-12/checkpoints/model_check.pkl \
--pretrained_model_name_or_path /data/pretrained/stable-diffusion-2-1-base \
--ram_ft_path /data/pretrained/DAPE.pth \
--ram_path /data/pretrained/ram_swin_large_14m.pth