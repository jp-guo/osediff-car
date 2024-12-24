export CUDA_VISIBLE_DEVICES=4
python main_test_qf_osediff.py \
-i testsets/LIVE1_color \
-o test_results/LIVE1_color_QF_OSEDIFF \
--osediff_path /home/guojinpei/diff-car/training_results/train_diff_512_div2k+flickr_fix_qf/2024-12-19_22-30-26/checkpoints/model_max_train_steps.pkl \
--qf_encoder_path /home/guojinpei/diff-car/training_results/train_qf_oesdiff_fix_qf/2024-12-21_20-21-16/checkpoints/model.pkl \
--pretrained_model_name_or_path /data/pretrained/stable-diffusion-2-1-base \
--ram_ft_path /data/pretrained/DAPE.pth \
--ram_path /data/pretrained/ram_swin_large_14m.pth