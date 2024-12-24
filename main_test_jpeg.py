import os.path
import logging

import numpy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import cv2
from utils import utils_logger
from utils import utils_image as util
import pyiqa


def main():
    quality_factor_list = [1, 5, 10, 20, 30, 40]    # 5, 10, 20, 30, 40
    testset_name = 'LIVE1_color'  # 'LIVE1_color' 'BSDS500_color' 'ICB', 'DIV2K_valid'
    n_channels = 3  # set 1 for grayscale image, set 3 for color image

    results = 'test_results'
    H_path = 'testsets/LIVE1_color'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    border = 0

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    dists_metric = pyiqa.create_metric('dists', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    for quality_factor in quality_factor_list:

        result_name = testset_name + '_' + 'jpeg'

        E_path = os.path.join(results, result_name, str(quality_factor))  # E_path, for Estimated images
        util.mkdir(E_path)

        logger_name = result_name + '_qf_' + str(quality_factor)
        utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info('--------------- quality factor: {:d} ---------------'.format(quality_factor))

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

        H_paths = util.get_image_paths(H_path)
        for idx, img in tqdm(enumerate(H_paths)):

            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d}--> {:>10s}'.format(idx + 1, img_name + ext))
            img_L = util.imread_uint(img, n_channels=n_channels)

            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            img_L = util.uint2tensor4(img_L)
            img_E = img_L.to(device)

            img_E = util.tensor2single(img_E)
            img_E = util.single2uint(img_E)
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()
            # --------------------------------
            # PSNR and SSIM, PSNRB
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            psnrb = util.calculate_psnrb(img_H, img_E, border=border)

            util.imsave(img_E, os.path.join(E_path, img_name + '.png'))
            if img_E.ndim == 3:
                img_E, img_H = torch.tensor(img_E, device=device).permute(2, 0, 1).unsqueeze(0), torch.tensor(img_H, device=device).permute(2, 0, 1).unsqueeze(0)
            else:
                img_E, img_H = torch.tensor(img_E, device=device).unsqueeze(0).unsqueeze(0), torch.tensor(img_H, device=device).unsqueeze(0).unsqueeze(0)
            img_E, img_H = img_E / 255., img_H / 255.
            lpips = lpips_metric(img_E, img_H)
            dists = dists_metric(img_E, img_H)
            niqe = niqe_metric(img_E, img_H)
            musiq = musiq_metric(img_E, img_H)
            maniqa = maniqa_metric(img_E, img_H)
            clipiqa = clipiqa_metric(img_E, img_H)

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnrb'].append(psnrb)
            test_results['lpips'].append(lpips.item())
            test_results['dists'].append(dists.item())
            test_results['niqe'].append(niqe.item())
            test_results['musiq'].append(musiq.item())
            test_results['maniqa'].append(maniqa.item())
            test_results['clipiqa'].append(clipiqa.item())

            # logger.info(
            #     '{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name + ext, psnr, ssim, psnrb))
            # logger.info('predicted quality factor: {:d}'.format(round(float(QF * 100))))

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
        avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        avg_dists = sum(test_results['dists']) / len(test_results['dists'])
        avg_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
        avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
        avg_maniqa = sum(test_results['maniqa']) / len(test_results['maniqa'])
        avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])

        logger.info(
            'Average PSNR/SSIM/PSNRB/LPIPIS/DISTS/NIQE/MUSIQ/MANIQA/CLIPIQA - {} -: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                result_name + '_' + str(quality_factor), ave_psnr, ave_ssim, ave_psnrb, avg_lpips, avg_dists,avg_niqe, avg_musiq, avg_maniqa, avg_clipiqa))

        with open(os.path.join(results, result_name, 'results.csv'), 'a') as f:
            print(quality_factor, ave_psnr, ave_ssim, ave_psnrb, avg_lpips, avg_dists, avg_niqe, avg_musiq, avg_maniqa, avg_clipiqa, sep=',', end='\n', file=f)

if __name__ == '__main__':
    main()
