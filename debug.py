import os
import sys

import utils.utils_image as utils

sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import cv2
from collections import OrderedDict

from diffusion.qf_osediff import QF_OSEDiff
from diffusion.osediff import OSEDiff_gen
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
import pyiqa
from tqdm import tqdm


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    lq_ram = ram_transforms(lq).to(dtype=dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"

    return validation_prompt, lq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input',
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output',
                        help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=128)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument('--train_decoder', action='store_true')

    args = parser.parse_args()

    # initialize the model
    model = QF_OSEDiff(args)
    # model = OSEDiff_gen(args)

    # get ram model
    DAPE = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # set weight type
    DAPE = DAPE.to(dtype=weight_dtype)

    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    H_paths = utils.get_image_paths(args.input_image)
    print(f'There are {len(H_paths)} images.')

    device = 'cuda'
    # lpips_metric = pyiqa.create_metric('lpips', device=device)
    # dists_metric = pyiqa.create_metric('dists', device=device)
    # niqe_metric = pyiqa.create_metric('niqe', device=device)
    # musiq_metric = pyiqa.create_metric('musiq', device=device)
    # maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    # clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    # f = open(os.path.join(args.output_dir, 'results.csv'), 'a')
    for quality_factor in [10]:      # 5, 10, 20, 30, 40
        os.makedirs(os.path.join(args.output_dir, str(quality_factor)), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, str(quality_factor)+'_ori'), exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnrb'] = []
        test_results['lpips'] = []
        test_results['dists'] = []
        test_results['niqe'] = []
        test_results['musiq'] = []
        test_results['maniqa'] = []
        test_results['clipiqa'] = []

        cnt = 0
        for idx, img in tqdm(enumerate(H_paths)):
            img_name, ext = os.path.splitext(os.path.basename(img))

            img_H = Image.open(img).convert('RGB')

            # vae can only process images with height and width multiples of 8
            new_width = img_H.width - img_H.width % 8
            new_height = img_H.height - img_H.height % 8
            img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

            img_L = img_H.copy()

            img_L = np.array(img_L)
            n_channels = img_L.shape[-1]
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

            img_L = Image.fromarray(img_L)
            # get caption
            validation_prompt, lq = get_validation_prompt(args, img_L, DAPE)

            # translate the image
            lq = lq * 2 - 1

            decoded_path = os.path.join('/home/guojinpei/diff-car/test_results/LIVE1_color_fbcnn_color', str(quality_factor),
                                        os.path.splitext(os.path.basename(img))[0] + '.png')
            img_model_decode = Image.open(decoded_path).convert('RGB')
            img_model_decode = img_model_decode.resize((new_width, new_height), Image.LANCZOS)

            # blur_path = os.path.join(f'/home/guojinpei/diff-car/testsets/LIVE1_color_{quality_factor}_blur',
            #                             os.path.splitext(os.path.basename(img))[0] + '.png')
            # img_blur = Image.open(blur_path).convert('RGB')
            # img_blur = img_blur.resize((new_width, new_height), Image.LANCZOS)

            # _, hq = get_validation_prompt(args, img_L, DAPE)
            _, hq = get_validation_prompt(args, img_model_decode, DAPE)
            # _, hq = get_validation_prompt(args, img_blur, DAPE)

            hq = hq * 2 - 1
            # img_E = model.guided_forward(lq, prompt=validation_prompt, hq=hq, loss_type='mse', alpha=0.005, bp=3)
            img_E = model(lq, prompt=validation_prompt, qf=torch.tensor(quality_factor / 100., device=device).reshape(-1, 1))
            # img_E = model(lq, prompt=validation_prompt)
            loss = ((img_E - hq) ** 2).sum()
            loss.backward()

        #     img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
        #     if args.align_method == 'adain':
        #         img_E = adain_color_fix(target=img_E, source=img_L)
        #     elif args.align_method == 'wavelet':
        #         img_E = wavelet_color_fix(target=img_E, source=img_L)
        #     else:
        #         pass
        #
        #     img_H = np.array(img_H)
        #     img_E = np.array(img_E)
        #
        #     psnr = utils.calculate_psnr(img_E, img_H, border=0)
        #     # print(psnr)
        #     ####
        #     # new_img_E = model.guided_forward(lq, prompt=validation_prompt, hq=hq, loss_type='ssim', alpha=0.5, bp=3)
        #     # new_img_E = transforms.ToPILImage()(new_img_E[0].cpu() * 0.5 + 0.5)
        #     # if args.align_method == 'adain':
        #     #     new_img_E = adain_color_fix(target=new_img_E, source=img_L)
        #     # elif args.align_method == 'wavelet':
        #     #     new_img_E = wavelet_color_fix(target=new_img_E, source=img_L)
        #     # else:
        #     #     pass
        #     # new_img_E = np.array(new_img_E)
        #     # new_psnr = utils.calculate_psnr(new_img_E, np.array(img_H), border=0)
        #     # print(f'{img}: {psnr}->{new_psnr}')
        #
        #     # ####
        #
        #     ssim = utils.calculate_ssim(img_E, img_H, border=0)
        #     psnrb = utils.calculate_psnrb(img_H, img_E, border=0)
        #
        #     utils.imsave(img_E, os.path.join(args.output_dir, str(quality_factor), img_name + '.png'))
        #     # utils.imsave(img_L, os.path.join(args.output_dir, str(quality_factor)+'_ori', img_name + '.png'))
        #
        #     img_E, img_H = torch.tensor(img_E, device=device).permute(2, 1, 0).unsqueeze(0), torch.tensor(img_H,
        #                                                                                                   device=device).permute(
        #         2, 1, 0).unsqueeze(0)
        #     img_E, img_H = img_E / 255., img_H / 255.
        #     lpips = lpips_metric(img_E, img_H)
        #     dists = dists_metric(img_E, img_H)
        #     niqe = niqe_metric(img_E, img_H)
        #     musiq = musiq_metric(img_E, img_H)
        #     maniqa = maniqa_metric(img_E, img_H)
        #     clipiqa = clipiqa_metric(img_E, img_H)
        #
        #     test_results['psnr'].append(psnr)
        #     test_results['ssim'].append(ssim)
        #     test_results['psnrb'].append(psnrb)
        #     test_results['lpips'].append(lpips.item())
        #     test_results['dists'].append(dists.item())
        #     test_results['niqe'].append(niqe.item())
        #     test_results['musiq'].append(musiq.item())
        #     test_results['maniqa'].append(maniqa.item())
        #     test_results['clipiqa'].append(clipiqa.item())
        #
        # ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        # ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        # ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
        # avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        # avg_dists = sum(test_results['dists']) / len(test_results['dists'])
        # avg_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
        # avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
        # avg_maniqa = sum(test_results['maniqa']) / len(test_results['maniqa'])
        # avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])
        #
        # print(
        #     'Average PSNR/SSIM/PSNRB/LPIPIS/DISTS/NIQE/MUSIQ/MANIQA/CLIPIQA - {} -: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
        #         str(quality_factor), ave_psnr, ave_ssim, ave_psnrb, avg_lpips, avg_dists, avg_niqe,
        #         avg_musiq, avg_maniqa, avg_clipiqa))
        #
        # print(quality_factor, ave_psnr, ave_ssim, ave_psnrb, avg_lpips, avg_dists, avg_niqe, avg_musiq, avg_maniqa,
        #       avg_clipiqa, sep=',', end='\n', file=f)

    # f.close()
