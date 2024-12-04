import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_image import imread_uint, calculate_ssim, calculate_psnr, calculate_psnrb
from tqdm import tqdm


H_path = 'testsets/LIVE1_color/bikes.bmp'
# H_path = 'testsets/Classic5/2.bmp'
n_channels = 3
img_H = imread_uint(H_path, n_channels)
ssim = []
psnr = []
psnrb = []

qs = range(1, 100)
for quality_factor in tqdm(qs):
    img_L = img_H.copy()
    if n_channels == 3:
        img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_L = cv2.imdecode(encimg, int(n_channels == 3))
    if n_channels == 3:
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    else:
        img_L = img_L[..., None]
    ssim.append(calculate_ssim(img_L, img_H))
plt.plot(qs, ssim)
plt.show()
