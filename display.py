import matplotlib.pyplot as plt
from PIL import Image
import os


fig, axes = plt.subplots(1, 5, figsize=(25, 5))

dataset = 'LIVE1_color'
models = ['naive_fbcnn_color_45000', 'fbcnn_color', 'OSEDIFF_no_reg', 'jpeg']
basename = 'caps.png'
img = Image.open(os.path.join('testsets', dataset, basename[:-4] + '.bmp'))
left = 100
upper = 50
right = 400
lower = 350
crop_area = (left, upper, right, lower)
img = img.crop(crop_area)
axes[0].imshow(img)
axes[0].axis('off')
axes[0].set_title('Original Image')

for i, ax in enumerate(axes[1:]):
    path = os.path.join('test_results', dataset + '_' + models[i], '20', basename)
    img = Image.open(path)
    img = img.crop(crop_area)
    ax.imshow(img)
    ax.set_title(models[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{dataset}_{basename}', bbox_inches='tight')