import cv2
from utils.utils_image import imread_uint
from tqdm import tqdm
import os
import numpy as np


H_path = 'testsets/LIVE1_color/bikes.bmp'
os.makedirs('noised', exist_ok=True)
# H_path = 'testsets/Classic5/2.bmp'
n_channels = 3
img_H = imread_uint(H_path, n_channels)

# qs = range(0, 101, 10)
# for quality_factor in tqdm(qs):
#     img_L = img_H.copy()
#     if n_channels == 3:
#         img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
#     result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
#     img_L = cv2.imdecode(encimg, int(n_channels == 3))
#     if n_channels == 3:
#         # img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
#         pass
#     else:
#         img_L = img_L[..., None]
#     # ssim.append(calculate_ssim(img_L, img_H))
#     output_path = f'noised/{os.path.basename(H_path)}_{quality_factor}.png'
#     cv2.imwrite(output_path, img_L)

img_L = img_H.copy()
if n_channels == 3:
    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
img_L = cv2.imdecode(encimg, int(n_channels == 3))
if n_channels == 3:
    # img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    pass
else:
    img_L = img_L[..., None]

img_LL = img_L.copy()
if n_channels == 3:
    img_LL = cv2.cvtColor(img_LL, cv2.COLOR_RGB2BGR)
result, encimg = cv2.imencode('.jpg', img_LL, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
img_LL = cv2.imdecode(encimg, int(n_channels == 3))
if n_channels == 3:
    # img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    pass
else:
    img_LL = img_LL[..., None]

print(np.abs(img_L - img_LL).sum())
