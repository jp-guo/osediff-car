from flask import Flask, render_template_string, send_from_directory
import os
import shutil
from itertools import groupby

app = Flask(__name__)

# HTML 模板，用于展示图片
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片展示</title>
    <style>
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            padding: 20px;
        }
        .image-item {
            text-align: center;
            flex: 0 0 30%
        }
        .image-item img {
            # max-width: 200px;
            width: 100%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1 style="text-align:center;">图片展示</h1>
    <div class="container">
        {% for group, filenames in grouped_images.items() %}
            {% for filename in filenames %}
                <div class="image-item">
                    <img src="{{ url_for('serve_image', filename=filename) }}" alt="{{ filename }}">
                    <p>{{ filename }}</p>
                </div>
            {% endfor %}
        {% endfor %}
    </div>
</body>
</html>
"""

# 这里设置图片存储的文件夹路径
IMAGE_FOLDER = "web"

def group_images(images):
    # 按文件名前缀进行排序
    images.sort(key=lambda x: x.split('_'))  # 假设用下划线作为分隔符，取前缀
    # 按前缀分组
    grouped_images = {key: list(group) for key, group in groupby(images, key=lambda x: x.split('_')[0])}
    return grouped_images

@app.route('/')
def index():
    # 获取文件夹中的所有图片文件（jpg, png, jpeg等）
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # return render_template_string(html_template, images=images)
    grouped_images = group_images(images)
    return render_template_string(html_template, grouped_images=grouped_images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    testname = "Urban100"
    qf = str(5)
    roots = [os.path.join(f'/home/guojinpei/car-baselines/PromptCIR/test_results/{testname}', qf),
             os.path.join(f'/home/guojinpei/car-baselines/SUPIR/test_results/{testname}', qf),
             os.path.join(f'/home/guojinpei/car-baselines/DiffBIR/results/custom_{testname}_{qf}'),
             os.path.join(f'/home/guojinpei/diff-car/test_results/{testname}_OSEDIFF_no_reg', qf),
             # os.path.join(f'/home/guojinpei/diff-car/test_results/{testname}_jpeg', qf),
             # f'/home/guojinpei/diff-car/testsets/{testname}']
             os.path.join(f'/data/dataset/CAR/{testname}_{qf}'),
             os.path.join(f'/data/dataset/CAR/{testname}')
             ]
    models = ['PromptCIR','SUPIR', 'DiffBIR', 'ours', 'JPEG', 'GT']
    for model, root in zip(models, roots):
        for img in os.listdir(root):
            img_name, ext = os.path.splitext(img)
            shutil.copy(os.path.join(root, img), os.path.join(IMAGE_FOLDER, img_name + '_' + model + ext))

    # 启动Flask应用
    app.run(host='0.0.0.0', port=3888, debug=True)
    try:
        shutil.rmtree(IMAGE_FOLDER)
    except Exception:
        pass
