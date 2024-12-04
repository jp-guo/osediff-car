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

from diffusion.osediff import OSEDiff_test
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from utils import utils_option as option


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

    # parser.add_argument("--datasets", default='options/train_diff_color.json')

    args = parser.parse_args()

    # initialize the model
    model = OSEDiff_test(args)

    # args.datasets = option.parse_dataset(args.datasets)['datasets']
    # for phase, dataset_opt in args.datasets.items():
    #     if phase == 'test':
    #         test_set = define_Dataset(dataset_opt)
    #         test_set.normalize = True
    #         # print('Dataset [{:s} - {:s}] is created.'.format(test_set.__class__.__name__, dataset_opt['name']))
    #         dl_test = DataLoader(test_set, batch_size=1,
    #                              shuffle=False, num_workers=1,
    #                              drop_last=False, pin_memory=True)
    #     else:
    #         raise NotImplementedError("Phase [%s] is not recognized." % phase)


    # get all input images
    # if os.path.isdir(args.input_image):
    #     image_names = sorted(glob.glob(f'{args.input_image}/*.jpg'))
    # else:
    #     image_names = [args.input_image]

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
    quality_factor = 10
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnrb'] = []
    cnt = 0
    for idx, img in enumerate(H_paths):
        img_name, ext = os.path.splitext(os.path.basename(img))

        img_H = Image.open(img).convert('RGB')

        # vae can only process images with height and width multiples of 8
        new_width = img_H.width - img_H.width % 8
        new_height = img_H.height - img_H.height % 8
        img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

        img_L = img_H.copy()

        # img_L = utils.imread_uint(img, n_channels=n_channels)
        img_L = np.array(img_L)
        n_channels = img_L.shape[-1]
        if n_channels == 3:
            img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
        _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
        if n_channels == 3:
            img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

        # img_L = utils.uint2tensor4(img_L)
        # img_L = img_L.to("cuda")

        # input_image = batch["L"].to("cuda")
        # make sure that the input image is a multiple of 8
        # input_image = Image.open(image_name).convert('RGB')
        # ori_width, ori_height, _ = img_L.shape
        # rscale = args.upscale
        # resize_flag = False
        # if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        #     scale = (args.process_size // rscale) / min(ori_width, ori_height)
        #     img_L = img_L.resize((int(scale * ori_width), int(scale * ori_height)))
        #     resize_flag = True
        # img_L = Image.fromarray(img_L).resize((img_L.shape[0] * rscale, img_L.shape[1] * rscale))
        #
        # new_width = img_L.width - img_L.width % 8
        # new_height = img_L.height - img_L.height % 8
        # img_L = img_L.resize((new_width, new_height), Image.LANCZOS)
        # bname = os.path.basename(image_name)

        img_L = Image.fromarray(img_L)
        # get caption
        validation_prompt, lq = get_validation_prompt(args, img_L, DAPE)
        # if args.save_prompts:
        #     txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
        #     with open(txt_save_path, 'w', encoding='utf-8') as f:
        #         f.write(validation_prompt)
        #         f.close()
        # print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        # translate the image
        with torch.no_grad():
            lq = lq * 2 - 1
            img_E = model(lq, prompt=validation_prompt)
            img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                img_E = adain_color_fix(target=img_E, source=img_L)
            elif args.align_method == 'wavelet':
                img_E = wavelet_color_fix(target=img_E, source=img_L)
            else:
                pass
            # if resize_flag:
            #     output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))

        img_H = np.array(img_H)
        img_E = np.array(img_E)
        # breakpoint()
        # img_E = utils.tensor2single(img_E)
        # img_E = utils.single2uint(img_E)

        # output_pil.save(os.path.join(args.output_dir, bname))

        try:
            psnr = utils.calculate_psnr(img_E, img_H, border=0)
            ssim = utils.calculate_ssim(img_E, img_H, border=0)
            psnrb = utils.calculate_psnrb(img_H, img_E, border=0)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnrb'].append(psnrb)
            # print(
            #     '{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name + ext, psnr, ssim, psnrb))
        except Exception:
            print(f'Error: {img_E.shape} != {img_H.shape}')
            cnt += 1
        # logger.info('predicted quality factor: {:d}'.format(round(float(QF * 100))))

        # util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
        utils.imsave(img_E, os.path.join(args.output_dir, img_name + '.png'))
    # print(f'Failed case: {cnt}')
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
    print(
        'Average PSNR/SSIM/PSNRB - {} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'.format(
            str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))

