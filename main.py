# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""

import sys
import time
import getopt
import os
import numpy as np
from PIL import Image as PILImage
import cv2
import dlib
import bottleneck as bn
from scipy.cluster.vq import *
from scipy.misc import imresize
from pylab import *
import style
sys.path.append('./Face-Pose-Net/')
import main_fpn

# Path of the Caffe installation.
_CAFFE_ROOT =  "/usr/local/Cellar/crfasrnn/caffe/"

# Model definition and model file paths
_MODEL_DEF_FILE = "TVG_CRFRNN_new_deploy.prototxt"  # Contains the network definition
_MODEL_FILE = "TVG_CRFRNN_COCO_VOC.caffemodel"  # Contains the trained weights. Download from http://goo.gl/j7PrPZ

sys.path.insert(0, _CAFFE_ROOT + "python")
import caffe

_MAX_DIM = 500
### 各个部分的颜色设置以及参数设置
RGB_color_hair  = [255,97,3]
RGB_color_eye   = [255,190,190]
RGB_color_upmouth = [255,190,191]
RGB_color_downmouth = [205,100,29]
RGB_color_pupil = [10,10,10]   #瞳孔颜色
outline_degree = 100 #范围为0-255 灰度
###以上为参数设置

def compute_y(a, b, scope_y, n, s):
    if n==0:
        y = [b[0,0] - a[0,0], a[0,1] - b[0,1]]
        if a[0,1]>b[0,1]:
            k = (2**y[1])/float(y[0]+1)
            for i in range(a[0,0],b[0,0]):
                x = i-a[0,0]
                scope_y[i-s,n] = a[0,1]-log(k*(x+1))
        elif a[0,1]==b[0,1]:
            for i in range(a[0,0],b[0,0]):
                scope_y[i-s,n]=a[0,1]
        else:
            k = float(y[1])/y[0]**2
            for i in range(a[0,0],b[0,0]):
                x = i-a[0,0]
                scope_y[i-s,n] = a[0,1]-k*(x**2)
    else:
        if a[0,1]<b[0,1]:
            y = [a[0,0]-b[0,0], b[0,1]-a[0,1]]
            k = y[1]/y[0]**2
            for i in range(a[0,0],b[0,0]):
                x = i-b[0,0]
                scope_y[i-s,n] = b[0,1]-k*(x**2)
        elif a[0,1]==b[0,1]:
            for i in range(a[0,0],b[0,0]):
                scope_y[i-s,n] = a[0,1]
        else:
            y = [b[0,0]-a[0,0], a[0,1]-b[0,1]]
            k = y[1] / y[0] ** 2
            for i in range(a[0,0],b[0,0]):
                x = i-a[0,0]
                scope_y[i-s,n] = a[0,1]-k*(x**2)

    return scope_y

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.

    Args:
        num_cls: Number of classes

    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    """ Returns the segmentation of the given image.

    Args:
        model_def_file: File path of the Caffe model definition prototxt file
        model_file: File path of the trained model file (contains trained weights)
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
        inputs: List of images to be segmented

    Returns:
        The segmented image
    """

    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."

    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.

    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    # PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B]
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)


def resize_image(image):
    """ Resizes the image so that the largest dimension is not larger than 500 pixels.
        If the image's largest dimension is already less than 500, no changes are made.

    Args:
        Input image

    Returns:
        Resized image where the largest dimension is less than 500 pixels
    """

    width, height = image.shape[0], image.shape[1]
    max_dim = max(width, height)

    if max_dim > _MAX_DIM:
        if height > width:
            ratio = float(_MAX_DIM) / height
        else:
            ratio = float(_MAX_DIM) / width
        image = PILImage.fromarray(np.uint8(image))
        image = image.resize((int(height * ratio), int(width * ratio)), resample=PILImage.BILINEAR)
        image = np.array(image)

    return image

# 根据眼睛的关键点来画出眼睛轮廓
def checkEye(img, landmarks):
    #左眼关键点
    a = landmarks[36]
    b = landmarks[37]
    c = landmarks[38]
    f = landmarks[39]
    e = landmarks[40]
    d = landmarks[41]
    w = f[0,0]-a[0,0]
    s = a[0,0]
    scope_y = np.zeros((w,2))
    scope_y = compute_y(a, b, scope_y, 0, s)
    scope_y = compute_y(b, c, scope_y, 0, s)
    scope_y = compute_y(c, f, scope_y, 0, s)
    scope_y = compute_y(a, d, scope_y, 1, s)
    scope_y = compute_y(d, e, scope_y, 1, s)
    scope_y = compute_y(e, f, scope_y, 1, s)
    for i in range(a[0,0],f[0,0]):
        x = i-a[0,0]
        for j in range(int(scope_y[x,0]),int(scope_y[x,1])):
            img[j,i] = RGB_color_eye
    for i in range(b[0,0], c[0,0]):
        x = i-a[0,0]
        for j in range(int(scope_y[x,0]),int(scope_y[x,1])):
            img[j,i] = RGB_color_pupil

    #右眼关键点
    a = landmarks[42]
    b = landmarks[43]
    c = landmarks[44]
    f = landmarks[45]
    e = landmarks[46]
    d = landmarks[47]
    w = f[0, 0] - a[0, 0]
    s = a[0, 0]
    scope_y = np.zeros((w, 2))
    #调用compute_y函数画两点间平滑线以及返回scope_y的纵坐标y范围
    scope_y = compute_y(a, b, scope_y, 0, s)
    scope_y = compute_y(b, c, scope_y, 0, s)
    scope_y = compute_y(c, f, scope_y, 0, s)
    scope_y = compute_y(a, d, scope_y, 1, s)
    scope_y = compute_y(d, e, scope_y, 1, s)
    scope_y = compute_y(e, f, scope_y, 1, s)
    for i in range(a[0, 0], f[0, 0]):
        x = i - a[0, 0]
        for j in range(int(scope_y[x, 0]), int(scope_y[x, 1])):
            img[j, i] = RGB_color_eye
    for i in range(b[0,0], c[0,0]):
        x = i-a[0,0]
        for j in range(int(scope_y[x,0]),int(scope_y[x,1])):
            img[j,i] = RGB_color_pupil

    return img

# 根据嘴唇的关键点来画出嘴唇轮廓
def checkMouth(img, landmarks, pos1, pos2):
    #检查上嘴唇
    a = landmarks[48]
    b = landmarks[49]
    c = landmarks[50]
    d = landmarks[51]
    e = landmarks[52]
    f = landmarks[53]
    g = landmarks[54]
    h = landmarks[65]
    i = landmarks[66]
    j = landmarks[67]
    w = g[0, 0] - a[0, 0]
    s = a[0, 0]
    scope_y = np.zeros((w, 2))
    scope_y = compute_y(a, b, scope_y, 0, s)
    scope_y = compute_y(b, c, scope_y, 0, s)
    scope_y = compute_y(c, d, scope_y, 0, s)
    scope_y = compute_y(d, e, scope_y, 0, s)
    scope_y = compute_y(e, f, scope_y, 0, s)
    scope_y = compute_y(f, g, scope_y, 0, s)
    scope_y = compute_y(a, g, scope_y, 1, s)
    scope_y = compute_y(g, i, scope_y, 1, s)
    scope_y = compute_y(i, h, scope_y, 1, s)
    scope_y = compute_y(h, j, scope_y, 1, s)
    for i in range(a[0, 0], g[0, 0]):
        x = i - a[0, 0]
        for j in range(int32(scope_y[x, 0]), int32(scope_y[x, 1])):
            pos1.append((j,i))
    
    #检查下嘴唇
    a = landmarks[48]
    b = landmarks[61]
    c = landmarks[62]
    d = landmarks[63]
    e = landmarks[64]
    f = landmarks[54]
    g = landmarks[59]
    h = landmarks[58]
    i = landmarks[57]
    j = landmarks[56]
    k = landmarks[55]
    w = f[0, 0] - a[0, 0]
    s = a[0, 0]
    scope_y = np.zeros((w, 2))
    scope_y = compute_y(a, b, scope_y, 0, s)
    scope_y = compute_y(b, c, scope_y, 0, s)
    scope_y = compute_y(c, d, scope_y, 0, s)
    scope_y = compute_y(d, e, scope_y, 0, s)
    scope_y = compute_y(e, f, scope_y, 0, s)
    scope_y = compute_y(a, g, scope_y, 1, s)
    scope_y = compute_y(g, h, scope_y, 1, s)
    scope_y = compute_y(h, i, scope_y, 1, s)
    scope_y = compute_y(i, j, scope_y, 1, s)
    scope_y = compute_y(j, k, scope_y, 1, s)
    scope_y = compute_y(k, f, scope_y, 1, s)
    for i in range(a[0, 0], f[0, 0]):
        x = i - a[0, 0]
        for j in range(int32(scope_y[x, 0]), int32(scope_y[x, 1])):
            pos2.append((j, i))

    return img, pos1, pos2

def remove_noise(img):
    [w, h, k] = img.shape;
    # 去除人物分割后的毛边
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            temp = img[i, j]
            if ((temp[0] > 240 and temp[1] > 220 and temp[2] > 220) or (
                        temp[1] > 240 and temp[0] > 220 and temp[2] > 220) \
                        or (temp[2] > 240 and temp[0] > 220 and temp[1] > 220)):
                img[i, j] = [255, 255, 255]
    return img

def remove_white_noise(img):
    [w, h, k] = img.shape;
    # 去除人物分割后的毛边
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            temp = img[i, j]
            if ((temp[0] > 230 and temp[1] > 230 and temp[2] > 230) or (
                        temp[1] > 230 and temp[0] > 230 and temp[2] > 230) \
                        or (temp[2] > 230 and temp[0] > 230 and temp[1] > 230)):
                img[i, j] = [255, 255, 255]
    return img

def removeBG(input_file, input_name):
    output_file = "mid_result/res.png"
    gpu_device = -1  # Use -1 to run only on the CPU, use 0-3[7] to run on the GPU

    if gpu_device >= 0:
        print("GPU device ID: {}".format(gpu_device))
    else:
        print("Using the CPU (set parameters appropriately to use the GPU)")
    run_crfrnn(input_file, output_file, gpu_device)

    #remove background
    im = PILImage.open(input_file)
    original_im = np.array(im, dtype=np.uint8)
    im_done = PILImage.open(output_file)
    (x,y) = im.size
    im_done = im_done.resize((x,y),PILImage.ANTIALIAS)
    im_done = np.array(im_done, dtype=np.uint8)
    res = np.zeros(original_im.shape, dtype='uint8')

    for i in range(0, original_im.shape[0]):
        for j in range(0, original_im.shape[1]):
            temp = im_done[i,j]
            if (temp == 0):
                res[i, j] = [255, 255, 255]
            else:
                res[i, j] = original_im[i, j]

    img = PILImage.fromarray(np.uint8(res))
    save_name = 'mid_result/res_'+input_name+'.png'
    print save_name
    img.save(save_name)

def check_up_bound(x, y, w, img):
    for i in range(w):
        if(not (img[y,x+i,0]==255 and img[y,x+i,1]==255 and img[y,x+i,2]==255)):
            return False
    return True

def check_left_bound(x, y, h, img):
    for i in range(h):
        if(not (img[y+i,x,0]==255 and img[y+i,x,1]==255 and img[y+i,x,2]==255)):
            return False
    return True

def remove_left_face(x, y, KeyPoint, img):
    start = 0
    end = 0
    for i in range(KeyPoint[4][1]-y):
        row = y+i
        # 求当前行的起点位置
        if (row<=KeyPoint[5][1]):
            start = x
        elif (row > KeyPoint[5][1] and row <= KeyPoint[8][1]):
            start = KeyPoint[5][0] - (KeyPoint[5][1]-row) / ((KeyPoint[8][1] - KeyPoint[5][1]) / (KeyPoint[8][0] - KeyPoint[5][0]))
        elif (row > KeyPoint[8][1] and row <= KeyPoint[9][1]):
            start = KeyPoint[8][0] - (KeyPoint[8][1] - row) / ((KeyPoint[9][1] - KeyPoint[8][1]) / (KeyPoint[9][0] - KeyPoint[8][0]))
        elif (row > KeyPoint[9][1] and row <= KeyPoint[10][1]):
            start = KeyPoint[9][0] - (KeyPoint[9][1] - row) / ((KeyPoint[10][1] - KeyPoint[9][1]) / (KeyPoint[10][0] - KeyPoint[9][0]))
        elif (row > KeyPoint[10][1] and row <= KeyPoint[11][1]):
            start = KeyPoint[10][0] - (KeyPoint[10][1] - row) / ((KeyPoint[11][1] - KeyPoint[10][1]) / (KeyPoint[11][0] - KeyPoint[10][0]))
        elif (row > KeyPoint[11][1] and row <= KeyPoint[12][1]):
            start = KeyPoint[11][0] - (KeyPoint[11][1] - row) / ((KeyPoint[12][1] - KeyPoint[11][1]) / (KeyPoint[12][0] - KeyPoint[11][0]))
        else:
            start = KeyPoint[12][0] - (KeyPoint[12][1] - row) / ((KeyPoint[4][1] - KeyPoint[12][1]) / (KeyPoint[4][0] - KeyPoint[12][0]))
        #求当前行的终点位置
        if (row <= KeyPoint[1][1]):
            end = KeyPoint[1][0]
        elif (row > KeyPoint[1][1] and row <= KeyPoint[2][1]):
            end = KeyPoint[1][0] + (row-KeyPoint[1][1])/((KeyPoint[2][1]-KeyPoint[1][1])/(KeyPoint[2][0]-KeyPoint[1][0]))
        elif (row > KeyPoint[2][1] and row <= KeyPoint[3][1]):
            end = KeyPoint[2][0] + (row-KeyPoint[2][1])/((KeyPoint[3][1]-KeyPoint[2][1])/(KeyPoint[3][0]-KeyPoint[2][0]))
        elif (row > KeyPoint[3][1] and row <= KeyPoint[4][1]):
            end = KeyPoint[3][0] + (row-KeyPoint[3][1])/((KeyPoint[4][1]-KeyPoint[3][1])/(KeyPoint[4][0]-KeyPoint[3][0]))

        for j in range(end-start):
            img[row, start+j] = [255,255,255]

    return img

def fill_blank_face(KeyPoint, img_front):
    start_x = KeyPoint[4][0] - 1
    copy_x = KeyPoint[7][0] + 3
    copy_y = KeyPoint[7][1] - 4
    for row in range(KeyPoint[1][1], KeyPoint[4][1]):
        i = start_x
        copy_i = copy_x
        while (1):
            if (img_front[row,i,0]==255 and img_front[row,i,1]==255 and img_front[row,i,2]==255):
                img_front[row, i] = img_front[copy_y, copy_i]
                i += 1
                copy_i += 1
            else:
                break

        copy_y = copy_y + 1
    return img_front

def add_mouth(img, img_original, pos1, pos2):
    for k in range(len(pos1)):
        j = pos1[k][0]
        i = pos1[k][1]
        img[j, i] = img_original[j,i]
    for k in range(len(pos2)):
        j = pos2[k][0]
        i = pos2[k][1]
        img[j, i] = img_original[j,i]
    return img

def second_step(input_name, rate):
    print 'Load Object Cascode Classifier'
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 导入源图片
    img_front = cv2.imread('mid_result/res_'+input_name+'.png')
    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
    imgGray_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2GRAY)
    img_original = cv2.imread('mid_result/res_'+input_name+'.png')
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_profile = cv2.imread('./mid_result/profiles/profile_'+input_name+'.png')
    #  根据正脸框的大小resize侧脸大小
    img_profile = cv2.resize(img_profile, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    img_profile = cv2.cvtColor(img_profile, cv2.COLOR_BGR2RGB)
    #imgGray_profile = cv2.cvtColor(img_profile, cv2.COLOR_RGB2GRAY)
    print "######", rate

    # 去除分割背景后的毛边
    img_front = remove_noise(img_front)

    #img_profile 对侧脸照片进行预处理，清除类白色像素块
    img_profile = remove_white_noise(img_profile)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(img_front,1)
    KeyPoint = [[0 for i in range(2)] for j in range(15)]
    for n in range(len(rects)):
        #关键点存放在landmarks数组中
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img_front,rects[n]).parts()])
        KeyPoint[1] = [landmarks[27,0], landmarks[27,1]]
        KeyPoint[2] = [landmarks[35,0], landmarks[35,1]]
        KeyPoint[3] = [landmarks[54,0], landmarks[54,1]]
        KeyPoint[4] = [landmarks[8, 0], landmarks[8, 1]]
        KeyPoint[5] = [landmarks[3, 0], landmarks[3, 1]]
        KeyPoint[6] = [landmarks[22, 0], landmarks[22, 1]]
        KeyPoint[7] = [landmarks[42, 0], landmarks[42, 1]]
        KeyPoint[8] = [landmarks[4, 0], landmarks[4, 1]]
        KeyPoint[9] = [landmarks[5, 0], landmarks[5, 1]]
        KeyPoint[10] = [landmarks[6, 0], landmarks[6, 1]]
        KeyPoint[11] = [landmarks[7, 0], landmarks[7, 1]]

        # 检测出眼睛，并且随意改变颜色
        img_front = checkEye(img_front, landmarks)
        pos1 = []
        pos2 = []
        img_front, pos1, pos2 = checkMouth(img_front, landmarks, pos1, pos2)
    
    #正面照片人脸轮廓识别
    faces = faceCascade.detectMultiScale(imgGray_front, scaleFactor=1.3, minNeighbors=4, minSize=(60, 60), maxSize=(350, 350))
    x_left_up = 0
    y_left_up = 0
    for (x, y, w, h) in faces:
        # 调整上边框
        t = 0
        while(t==0):
            if(y != 0):
                y = y-1
            else:
                break
            if(check_up_bound(x, y, w, img_front)):
                t = 1
        # 调整左边框
        t = 0
        while(t==0):
            if(x != 0):
                x = x-1
            else:
                break;
            k = KeyPoint[5][1]-y
            if(check_left_bound(x, y, k, img_front)):
                t = 1
        x_left_up = x
        y_left_up = y

    # 去掉左半边脸
    img_front = remove_left_face(x_left_up, y_left_up, KeyPoint, img_front)

    # 填充空白区域
    img_front = fill_blank_face(KeyPoint, img_front)

    # 添加侧脸
    img_front = add_profile_face(KeyPoint, img_profile, img_front)

    # 模糊图片的连接边界
    #img_front = blur_edge(KeyPoint, img_front)

    # Add Mouth
    img_front = add_mouth(img_front, img_original, pos1, pos2)

    img_front=cv2.medianBlur(img_front,3)
    im = PILImage.fromarray(np.uint8(img_front))
    #im.show()
    im.save('mid_result/res_'+input_name+'.png')

    return True

def add_profile_face(KeyPoint, img_profile, img_front):
    #找到上边界m
    size = img_profile.shape
    m = size[0]
    for i in range(size[1]):
        for j in range(size[0]):
            if(not(img_profile[j,i,0]>253 and img_profile[j,i,1]>253 and img_profile[j,i,2]>253)):
                if(i<m):
                    m=i
                break;
    #找到下边界n
    n = 0
    for i in range(size[1]):
        for j in range(size[0]):
            jj = size[0] - j - 1
            if(not(img_profile[jj,i,0]>253 and img_profile[jj,i,1]>253 and img_profile[jj,i,2]>253)):
                if(jj>n):
                    n = jj
                break;
    l = size[1]
    for i in range(n-m):
        for j in range(size[1]):
            if (not (img_profile[i, j, 0] > 253 and img_profile[i, j, 1] > 253 and img_profile[i, j, 2] > 253)):
                if (i < l):
                    l = i
                break;
    # r = 0
    # for i in range(n-m):
    #     for j in range(size[1]):
    #         jj = size[1] - j - 1
    #         if (not (img_profile[i, jj, 0] > 253 and img_profile[i, jj, 1] > 253 and img_profile[i, jj, 2] > 253)):
    #             if (jj > r):
    #                 r = jj
    #             break;
    # for i in range(n-m):
    #     ii = m + i
    #     for j in range(r-l):
    #         jj = l + j
    #         img_front[ii, jj] = img_profile[ii, jj]
    # 把侧脸加入到正脸照片中,从下往上遍历
    start = n
    end = m
    level = KeyPoint[4][1]
    for i in range(n-m):
        row = start-i
        x = 0
        for j in range(size[1]):
            k = size[1] - j - 1
            if(not(img_profile[row,k,0]==255 and img_profile[row,k,1]==255 and img_profile[row,k,2]==255)):
                x = k
                break
        index = KeyPoint[6][0]
        while(not(img_profile[row,x,0]==255 and img_profile[row,x,1]==255 and img_profile[row,x,2]==255)):
            img_front[level, index] = img_profile[row, x]
            x -= 1
            index -= 1
        level -= 1

    return img_front

def edge_detect(input_name):
    img = cv2.imread('mid_result/res_'+input_name+'.png')
    img = image_sharpening(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    [w, h, k] = img.shape;

    sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)  # x方向的梯度
    sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)  # y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    res = np.zeros(img.shape, dtype='uint8') + 255
    for i in range(w - 1):
        for j in range(h - 1):
            if sobelCombined[i, j] > outline_degree:
                img[i, j] = [1, 0, 0];  # 人物的轮廓
    im = PILImage.fromarray(np.uint8(img))
    im.save('mid_result/res_'+input_name+'.png')
    return True

def coloring(input_name):
    img = cv2.imread('mid_result/res_'+input_name+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = np.zeros(img.shape, dtype='uint8')
    [w, h, k] = img.shape;

    # 人物分割色彩图
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            temp = img[i, j]
            if (temp[0] == RGB_color_hair[0] and temp[1] == RGB_color_hair[1] and temp[2] == RGB_color_hair[2]) \
                    or (temp[0] == RGB_color_eye[0] and temp[1] == RGB_color_eye[1] and temp[2] == RGB_color_eye[2]) \
                    or (temp[0] == RGB_color_upmouth[0] and temp[1] == RGB_color_upmouth[1] and temp[2] ==
                        RGB_color_upmouth[2]) \
                    or (temp[0] == RGB_color_downmouth[0] and temp[1] == RGB_color_downmouth[1] and temp[2] ==
                        RGB_color_downmouth[2]) \
                    or (temp[0] == RGB_color_pupil[0] and temp[1] == RGB_color_pupil[1] and temp[2] == RGB_color_pupil[
                        2]) \
                    or (temp[0] == 1 and temp[1] == 0 and temp[2] == 0):
                res[i, j] = temp
            else:
                if (temp[0] > 105 and temp[1] > 105 and temp[2] > 105 and
                                temp[0] - temp[2] > 15 and temp[0] - temp[1] > 15) \
                        or (temp[0] > 200 and temp[1] > 210 and temp[2] > 170 and
                                    abs(temp[0] - temp[2]) <= 15 and temp[0] > temp[2] and temp[1] > temp[2]):
                    res[i, j] = [255, 0, 0]
                elif temp[0] != 255 and temp[1] != 255 and temp[2] != 255:
                    res[i, j] = [0, 0, 255]

    img_bg = PILImage.open('Picasso/bg_picasso/background2_dealed.jpg')
    img_cloth = PILImage.open('Picasso/clothes.jpg')
    img_man = PILImage.open('Picasso/skin.jpg')
    img_left_face = PILImage.open('Picasso/face_skin.jpg')
    # reshape
    img_bg = img_bg.resize((h, w), PILImage.ANTIALIAS)
    img_cloth = img_cloth.resize((h, w), PILImage.ANTIALIAS)
    img_man = img_man.resize((h, w), PILImage.ANTIALIAS)
    img_left_face = img_left_face.resize((h, w), PILImage.ANTIALIAS)
    img_bg = np.array(img_bg, dtype=np.uint8)
    img_cloth = np.array(img_cloth, dtype=np.uint8)
    img_man = np.array(img_man, dtype=np.uint8)
    img_left_face = np.array(img_left_face, dtype=np.uint8)

    final_res = np.zeros(img.shape, dtype='uint8')
    for i in range(0, w - 1):
        for j in range(0, h - 1):
            temp = res[i, j]
            if temp[0] == 255 and temp[1] == 0 and temp[2] == 0:  # skin
                if i < 201 and j < 156:
                    final_res[i, j] = img_left_face[i, j]
                else:
                    final_res[i, j] = img_man[i, j]
            elif temp[0] == 0 and temp[1] == 0 and temp[2] == 255:  # clothes
                final_res[i, j] = img_cloth[i, j];
            elif temp[0] == 0 and temp[1] == 0 and temp[2] == 0:  # background
                final_res[i, j] = img_bg[i, j];
            else:
                final_res[i, j] = res[i, j]

    im = PILImage.fromarray(np.uint8(final_res))
    im.save('mid_result/res_'+input_name+'.png')

    return True

def style_transformation(input_name):
    style.style_transfer(input_name)
    return True

def generateFace(input_name):
    img = cv2.imread('mid_result/res_'+input_name+'.png')
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    file_name = 'Face-Pose-Net/input_image.csv'
    with open(file_name,'w') as file_object:
        file_object.write('ID,FILE,FACE_X,FACE_Y,FACE_WIDTH,FACE_HEIGHT\n')
        file_object.write('side,../mid_result/res_%s.png,%s,%s,%s,%s\n'%(input_name,str(x),str(y),str(w),str(h)))
    os.chdir('Face-Pose-Net')
    main_fpn.run()
    
    #process the generated profile face.
    img = cv2.imread('output_render/side/side_rendered_aug_-75_00_10.jpg')
    #min_value = 160
    #for i_test in range(180):
#	i = i_test + 10
#	for j_test in range(100):
#	    j = 160 - j_test
#	    if (not (img[i,j,0]>240 and img[i,j,1]>240 and img[i,j,2]>240)):
#		current_min = j
#		break
#	if (min_value > current_min):
#	    min_value = current_min

    img_face = img[10:190, 60:145]

    # case--black noise in left part
    hi = img_face.shape[0]
    wi = img_face.shape[1]
    for i in range(hi):
        for j in range(wi):
	    if(img_face[i,j,0]<5 and img_face[i,j,1]<5 and img_face[i,j,2]<5):
	        img_face[i,j] = [255,255,255]
 
    # 锐化侧脸 
    #img_face = image_sharpening(img_face)

    cv2.imwrite("../mid_result/profiles/profile_"+input_name+'.png', img_face)

    os.chdir('../')
    return h/160.0

def image_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imwrite('test_sharpening.jpg', dst)
    return dst

def main(argv):
    input_file = argv[0] #"images/qianwu2.jpg"
    s = input_file.find('/')
    input_name = input_file[s+1:-4]
    print input_name
    
    #根据需要是否锐化图片
    img = cv2.imread(input_file)
    image_sharpening(img)
    
    # the 1st step ----- remove the background
    removeBG(input_file, input_name)
    # generate side face
    scale_rate = generateFace(input_name) 
    
    # the 2nd step ----- 68 points detect and process based on this
    second_step(input_name, scale_rate)
    # the 3rd step ----- edge detect
    edge_detect(input_name)
    # the 4th step ----- coloring
    coloring(input_name)
    # the 5th step ----- ML-style transformation with CNN
    style_transformation(input_name)

if __name__ == "__main__":
    main(sys.argv[1:])

