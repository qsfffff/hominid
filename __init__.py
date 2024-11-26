import csv
import json
import os

import numpy as np
from PIL import Image
from flask import Flask, render_template, send_from_directory, request, Response

from flask_cors import *

import cv2 as cv

from matplotlib import pyplot as plt
from skimage import measure, filters, img_as_ubyte

from strUtil import pic_str

from fcm import get_centroids, get_label, get_init_fuzzy_mat, fcm
from lv_set import find_lsf, get_params

from preprocess import gamma_trans, clahe_trans

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50

from src import UNet

from src import deeplabv3_resnet50

from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN

from draw_box_utils_mg import draw_objs

# 配置Flask路由，使得前端可以访问服务器中的静态资源
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'tif'}

global src_img, pic_path, res_pic_path, message_get, pic_name, final


def allowed_file(file):
    return '.' in file and file.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def del_files(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 第一步：删除文件
        for t in files:
            os.remove(os.path.join(root, t))  # 删除文件
        # 第二步：删除空文件夹
        for t in dirs:
            os.rmdir(os.path.join(root, t))  # 删除一个空目录


# 主页
@app.route('/')
def hello():
    return render_template('main.html')
@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/example/')
def example_index():
    return render_template('example/example-datasets.html')
# 展示海马体exp
@app.route('/example/classic/hippocampus')
def show_example_hippocampus():
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    return render_template('example/classic/hippocampus-index.html', header=header, data1=data1, data2=data2,
                           data3=data3, data4=data4)


# 展示胸部X光exp
@app.route('/example/classic/chest')
def show_example_chest():
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/classic/chest-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6)


# 展示眼底血管exp
@app.route('/example/classic/eye')
def show_example_eye():
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/classic/eye-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6)


@app.route('/index_data_chest')
def line_stack_data_chest():
    data_list = {}
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7

    return Response(json.dumps(data_list), mimetype='application/json')


# eye_exp_data
@app.route('/index_data_eye')
def data_eye():
    data_list = {}
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7

    return Response(json.dumps(data_list), mimetype='application/json')


# hippocampus_exp_data
@app.route('/index_data_hippocampus')
def data_hippocampus():
    data_list = {}
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    return Response(json.dumps(data_list), mimetype='application/json')


# 展示不同网络对睑板腺数据集的分割效果
@app.route("/example/dl")
def dl_data():
    filename = 'static/csv/dl_data.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
    return render_template('example/dl/dl-index.html', data1=data1, data2=data2, data3=data3)


@app.route("/dl_data1")
def dl_data_m():
    data_list = {}
    filename = 'static/csv/dl_data1.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    data_list['header'] = header
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route("/dl_data2")
def dl_data_c():
    data_list = {}
    filename = 'static/csv/dl_data2.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    data_list['header'] = header
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    return Response(json.dumps(data_list), mimetype='application/json')


# 实时体验功能首页
@app.route('/live-index/')
def live_index():
    return render_template('/live/live-index.html')


# 图片上传相关
@app.route('/live-classic')
def upload_test():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-classic.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/live-classic/upload-success', methods=['POST'])
def upload_pic():
    del_files('static/tempPics')
    img1 = request.files['photo']
    if img1 and allowed_file(img1.filename):
        img = Image.open(img1.stream)

    # 保存图片
    global pic_path, res_pic_path
    # 为临时图片生成随机id
    pic_path = 'tempPics/' + pic_str().create_uuid() + '.png'
    img.save('static/' + pic_path)
    global src_img
    src_img = cv.imread('static/' + pic_path)
    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    # 预处理
    src_img = clahe_trans(src_img)
    src_img = gamma_trans(src_img)

    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-classic.html', pic_path=pic_path, res_pic_path=res_pic_path)


# 获取算法信息
@app.route('/live-classic/upload-success', methods=['GET'])
def get_Algorithm():
    global message_get
    message_get = str(request.values.get("algorithm"))


# 使用对应算法进行处理
@app.route('/live-classic/upload-success/result')
def algorithm_process():
    global src_img, res_pic_path, pic_path, message_get, pic_name
    if message_get == 'SOBEL':
        # 边缘检测之Sobel 算子
        edges = filters.sobel(src_img)
        # 浮点型转成uint8型
        edges = img_as_ubyte(edges)
        plt.figure()
        plt.imshow(edges, plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        pic_name = 'sobel.png'
        res_pic_path = 'tempPics/' + pic_name
        plt.savefig('static/' + res_pic_path)

    elif message_get == 'OTSU':
        _, otsu_img = cv.threshold(src_img, 0, 255, cv.THRESH_OTSU)
        pic_name = 'eye_otsu.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, otsu_img)

    elif message_get == 'WATERSHED':
        # 基于直方图的二值化处理
        _, thresh = cv.threshold(src_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # 做开操作，是为了除去白噪声
        kernel = np.ones((3, 3), dtype=np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # 做膨胀操作，是为了让前景漫延到背景，让确定的背景出现
        sure_bg = cv.dilate(opening, kernel, iterations=2)

        # 为了求得确定的前景，也就是注水处使用距离的方法转化
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        # 归一化所求的距离转换，转化范围是[0, 1]
        cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
        # 再次做二值化，得到确定的前景
        _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 得到不确定区域也就是边界所在区域，用确定的背景图减去确定的前景图
        unknow = cv.subtract(sure_bg, sure_fg)

        # 给确定的注水位置进行标上标签，背景图标为0，其他的区域由1开始按顺序进行标
        _, markers = cv.connectedComponents(sure_fg)

        # 让标签加1，这是因为在分水岭算法中，会将标签为0的区域当作边界区域（不确定区域）
        markers += 1

        # 是上面所求的不确定区域标上0
        markers[unknow == 255] = 0

        # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
        src_img = cv.cvtColor(src_img, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(src_img, markers)

        # 分水岭算法得到的边界点的像素值为-1
        src_img[markers == -1] = [0, 0, 255]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        pic_name = 'watershed.png'
        res_pic_path = 'tempPics/' + pic_name
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

    elif message_get == 'FCM':
        rows, cols = src_img.shape[:2]
        pixel_count = rows * cols
        image_array = src_img.reshape(1, pixel_count)

        # 初始模糊矩阵
        init_fuzzy_mat = get_init_fuzzy_mat(pixel_count)
        # 初始聚类中心
        init_centroids = get_centroids(image_array, init_fuzzy_mat)
        fuzzy_mat, centroids, target_function = fcm(init_fuzzy_mat, init_centroids, image_array)
        label = get_label(fuzzy_mat, image_array)
        fcm_img = label.reshape(rows, cols)
        pic_name = 'fcm.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, fcm_img)

    elif message_get == 'DRLSE':
        global final
        final = 0
        src_img = cv.resize(src_img, (128, 128))
        params = get_params(src_img)
        phi = find_lsf(**params)

        contours = measure.find_contours(phi, 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            final = contour

        ax.fill(final[:, 1], final[:, 0], color='w')
        ax.set_xticks([])
        ax.set_yticks([])

        pic_name = 'drlse.png'
        res_pic_path = 'tempPics/' + pic_name
        print(res_pic_path)
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

        del params

    return render_template('live/show.html', pic_path=pic_path, res_pic_path=res_pic_path, temp=message_get)


# 图片下载
@app.route('/live-classic/upload-success/result/download', methods=['GET'])
def download():
    global res_pic_path
    if request.method == "GET":
        path = 'static/tempPics'
        if path:
            return send_from_directory(path, pic_name, as_attachment=True)


# 获取dl算法信息
@app.route('/live-dl')
def upload_test1():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-dl.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/live-dl/upload-success', methods=['POST'])
def upload_pic1():
    del_files('static/tempPics')
    img1 = request.files['photo']
    if img1 and allowed_file(img1.filename):
        img = Image.open(img1.stream)
    # 保存图片
    global pic_path, res_pic_path
    # 为临时图片生成随机id
    pic_path = 'tempPics/' + pic_str().create_uuid() + '.png'
    img.save('static/' + pic_path)

    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-dl.html', pic_path=pic_path, res_pic_path=res_pic_path)

@app.route('/live-dl/upload-success', methods=['GET'])
def get_Algorithm1():
    global message_get
    message_get = str(request.values.get("algorithm"))
    return "Algorithm set successfully", 200  # 返回状态码200表示成功



@app.route('/live-dl/upload-success/result')
def algorithm_dl():
    global pic_path, message_get, pic_name, res_pic_path

    weights_path = f'static/assets/weights/{message_get}_best_model.pth'
    img_path = f'static/{pic_path}'
    num_classes = 2

    assert os.path.exists(weights_path), f"Weight file not found: {weights_path}"
    assert os.path.exists(img_path), f"图片文件未找到: {img_path}"

    weights_dict = torch.load(weights_path, map_location='cpu')
    if 'model' in weights_dict:
        weights_dict = weights_dict['model']  # 如果存在'model'键，则使用该键的值

    weights_path = 'static/assets/weights/' + message_get + '_best_model.pth'
    img_path = 'static/' + pic_path
    pic_name = message_get + '.png'
    print(message_get)

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if message_get == 'fcn':
        # create model
        model = fcn_resnet50(aux=False, num_classes=num_classes)
        # delete weights about aux_classifier
        weights_dict = torch.load(weights_path, map_location='cpu')
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']

        try:
            # 尝试加载权重
            model.load_state_dict(weights_dict)
        except RuntimeError as e:
            # 如果分类层大小不匹配，重新初始化分类层
            print(f"Warning: {e}")
            model.load_state_dict({k: v for k, v in weights_dict.items() if "classifier.4" not in k}, strict=False)
            with torch.no_grad():
                model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
                torch.nn.init.xavier_uniform_(model.classifier[4].weight)
                torch.nn.init.zeros_(model.classifier[4].bias)
        model.to(device)

    elif message_get == 'unet':
        # create model
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)

        # load weights
        weights_dict = torch.load(weights_path, map_location='cpu')
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']
        model.load_state_dict(weights_dict)
        model.to(device)

    elif message_get == 'deeplab':
        # create model
        model = deeplabv3_resnet50(aux=True, num_classes=num_classes)
        weights_dict = torch.load(weights_path, map_location='cpu')
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']
        model.load_state_dict(weights_dict)
        model.to(device)

    else:
        return f"未知模型类型: {message_get}"

    # 图像预处理
    original_img = Image.open(img_path).convert('RGB')
    data_transform = transforms.Compose([
        transforms.Resize(420),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = data_transform(original_img).unsqueeze(0).to(device)  # 扩展 batch 维度

    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(img)
        if message_get in ['fcn', 'deeplab']:
            prediction = output['out'].argmax(1).squeeze(0).to("cpu").numpy()
        elif message_get == 'unet':
            # prediction = output.argmax(dim=1).squeeze(0).to("cpu").numpy()
            if isinstance(output, dict):
                # 如果是字典类型，取主输出键 'out' 或类似键
                prediction = output['out'].argmax(dim=1).squeeze(0).to("cpu").numpy()
            else:
                # 如果是张量类型，直接处理
                prediction = output.argmax(dim=1).squeeze(0).to("cpu").numpy()

    # 将预测结果转换为图片
    prediction_img = (prediction * 255).astype('uint8')  # 将二分类掩码值放大到 [0, 255]
    mask = Image.fromarray(prediction_img)

    # 保存预测的图片结果
    res_pic_path = 'tempPics/' + pic_name
    mask.save('static/' + res_pic_path)
    return render_template('live/show-dl.html', pic_path=pic_path, res_pic_path=res_pic_path, temp=message_get)

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


# 图片下载
@app.route('/live-dl/upload-success/result/download', methods=['GET'])
def download1():
    global res_pic_path
    if request.method == "GET":
        path = 'static/tempPics'
        if path:
            return send_from_directory(path, pic_name, as_attachment=True)


if __name__ == '__main__':
    app.run(port=205, debug=True)
