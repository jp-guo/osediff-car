import os

import cv2
import matplotlib.pyplot as plt

root = 'testsets/LIVE1_color_1'
os.makedirs(root+'_blur', exist_ok=True)
for filename in os.listdir(root):
    image = cv2.imread(os.path.join(root, filename))
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=15, templateWindowSize=9, searchWindowSize=25)
    cv2.imwrite(os.path.join(root+'_blur', os.path.splitext(filename)[0] + '.png'), denoised_image)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    # plt.title('Denoised Image')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
