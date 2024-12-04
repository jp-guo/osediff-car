export CUDA_VISIBLE_DEVICES=2
python main_test_diff.py \
-i testsets/LIVE1_color \
-o testsets/LIVE1_color_output_finetune \
--osediff_path /zfsauton2/home/jinpeig/diff-car/experience/osediff/checkpoints/model_80001.pkl \
--pretrained_model_name_or_path /zfsauton2/home/jinpeig/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06 \
--ram_ft_path /zfsauton2/home/jinpeig/.cache/huggingface/hub/DAPE.pth \
--ram_path /zfsauton2/home/jinpeig/.cache/huggingface/hub/ram_swin_large_14m.pth