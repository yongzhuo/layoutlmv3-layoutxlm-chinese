# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/10/9 22:42
# @author  : Mo
# @function: 数据预处理-paddlepaddle-ocr


import traceback
import platform
import logging
import json
import os

from paddleocr import PaddleOCR, paddleocr
from typing import List
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2

from .configuration_extra import OCR_CUDA_VISIBLE_DEVICES

device = True if torch.cuda.is_available() and OCR_CUDA_VISIBLE_DEVICES != "-1" else False
paddleocr.logging.disable(logging.DEBUG)


if platform.system().lower() == "windows":
    ppocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=device)
else:
    try:
        ppocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=device)
    except Exception as e:
        ppocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False)
        print(traceback.print_exc())


def fill_image_with_border(img_or_path, std_size=224, img_idx=0, img_class="fill", path_img_out=None, is_draw=False, fill_color=215):
    """
        给小图填充边界框到指定尺寸(ocr小图用, 或者长宽悬殊的情况)
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        std_size[Int]: standard size of output image, eg. 320
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
    Returns:
        lines
    """
    # 读取图像
    flag = False
    if type(img_or_path)==str and os.path.exists(img_or_path):
        img = cv2.imdecode(np.fromfile(img_or_path, dtype=np.uint8), -1)  # 支持中文地址
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        flag = True
        img = img_or_path
    h, w, _ = img.shape
    border = [0, 0]
    # 图片填充边框到std_size尺寸
    if w < std_size or h < std_size:
        if h < std_size:
            border[0] = int((std_size - h) / 2.0)
        if w < std_size:
            border[1] = int((std_size - w) / 2.0)
        # 图像的上-下-左-右, 填充边界的长度
        img = cv2.copyMakeBorder(src=img, top=border[0], bottom=border[0], left=border[1], right=border[1],
                                 value=[fill_color, fill_color, fill_color],
                                 borderType=cv2.BORDER_CONSTANT
                                 )
    if is_draw:  # 保存填充后的图像
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img)  # 截取的图片
    return img


def ocr_paddle(img_or_path, std_size=224, len_limit=1):
    """  使用ocr-paddle  """
    # print(img_path)
    img = fill_image_with_border(img_or_path, std_size)
    res = ppocr.ocr(img, det=True, rec=True, cls=False)
    text_bboxs = []
    for idx, line in enumerate(res):
        bbox_4, text_score = line
        bbox_4_xs = [b[0] for b in bbox_4]
        bbox_4_ys = [b[1] for b in bbox_4]
        box_x1 = min(bbox_4_xs)
        box_y1 = min(bbox_4_ys)
        box_x2 = max(bbox_4_xs)
        box_y2 = max(bbox_4_ys)
        # box_x_center = (box_x1 + box_x2) / 2
        # box_y_center = (box_y1 + box_y2) / 2
        text = text_score[0]
        if not text.strip():
            continue
        text = "<s>{}</s>".format(text)
        text_bboxs.append((text, box_x1, box_y1, box_x2, box_y2))
    text_bboxs = merge_boxes(text_bboxs, len_limit=len_limit)
    texts, bboxs = [], []
    for t in text_bboxs:
        texts.append(t[0])
        bboxs.append((t[1], t[2], t[3], t[4]))
    return texts, bboxs


def merge_boxes(text_bboxs, len_limit=1):
    """ 框信息合并(正常每页只有单列的情况) """
    boxes = []
    # boxes-merge
    # res_page_sort = sorted(text_bboxs, key=lambda x: x[2])  # y0需要
    # len_page = len(res_page_sort)
    res_page_sort = text_bboxs
    len_page = len(res_page_sort)
    boxes_page = []
    box_s = []
    # 处理逻辑, 每行与之后的对比, 如果y0-jdx 与 y0-jdx+1的高差距在1以内, 就认为是同一行
    for jdx, r in enumerate(res_page_sort):
        if jdx < len_page - 1:  # 非最后一行
            x_center = (r[1] + r[3])/2
            y_center = (r[2] + r[4])/2
            x_center_next = (res_page_sort[jdx+1][1]+res_page_sort[jdx+1][3])/2
            y_center_next = (res_page_sort[jdx+1][2]+res_page_sort[jdx+1][4])/2
            if abs(y_center_next-y_center) <= len_limit:  # 合并, 以'开始'为分割
                r_next = res_page_sort[jdx + 1]
                if not box_s:
                    box_s.append(r)
                    box_s.append(r_next)
                else:
                    box_s.append(r_next)
            else:  # 不合并的时候就存储
                if box_s:
                    box_s_sort = sorted(box_s, key=lambda x: x[1])  # 按照x0排序
                    text = "".join([b[0] for b in box_s_sort])  # 左下右顶
                    x0 = min([b[1] for b in box_s_sort])
                    y0 = min([b[2] for b in box_s_sort])
                    x1 = max([b[3] for b in box_s_sort])
                    y1 = max([b[4] for b in box_s_sort])
                    box_s_merge = [text, x0, y0, x1, y1]
                    boxes_page.append(box_s_merge)
                    box_s = []
                else:
                    boxes_page.append(r)
        else:  # 最后一行
            if box_s:
                box_s_sort = sorted(box_s, key=lambda x: x[1])  # 按照x0排序
                text = "".join([b[0] for b in box_s_sort])  # 左下右顶
                x0 = min([b[1] for b in box_s_sort])
                y0 = min([b[2] for b in box_s_sort])
                x1 = max([b[3] for b in box_s_sort])
                y1 = max([b[4] for b in box_s_sort])
                box_s_merge = [text, x0, y0, x1, y1]
                boxes_page.append(box_s_merge)
                box_s = []
            else:
                boxes_page.append(r)
    return boxes_page[::-1]


def bbox_normalize(bbox, width, height):
    """ 标准化bbox """
    return [int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
            ]


def apply_ocr(img_or_path, img_size=224, max_len=510):
    """ 整个OCR流程 """
    # img = cv2.imdecode(np.fromfile(img_or_path, dtype=np.uint8), -1)  # 支持中文地址, shape-4
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imdecode(np.fromfile(img_or_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持中文地址, shape-3
    texts, bboxs = ocr_paddle(img)
    h, w, _ = img.shape
    bboxs_norm = []
    for bbox in bboxs:
        bboxs_norm.append(bbox_normalize(bbox, w, h))
    img_resize = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_resize = img_resize.transpose(2, 0, 1)
    return img_resize, texts[:max_len], bboxs_norm[:max_len]


def get_all_dirs_files(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    return list(set(path_files))


def save_json(jsons, json_path, indent=4):
    """
        保存json
    Args:
        path[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        jsons[Json]: json of input data, eg. [{"桂林": 132}]
        indent[int]: pretty-printed with that indent level, eg. 4
    Returns:
        None
    """
    with open(json_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False, indent=indent))
    fj.close()


def load_json(path, parse_int=None):
    """
        加载json
    Args:
        path_file[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        parse_int[Boolean]: equivalent to int(num_str), eg. True or False
    Returns:
        data[Any]
    """
    with open(path, mode="r", encoding="utf-8") as fj:
        model_json = json.load(fj, parse_int=parse_int)
    return model_json



if __name__ == '__main__':

    img_path = "zh_val_0.jpg"
    import numpy as np
    import cv2
    image = cv2.imread(img_path)
    b = np.array([[[100, 100], [250, 100], [300, 220], [100, 230]]], dtype=np.int32)
    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, b, 1, 255)
    cv2.fillPoly(im, b, 255)
    mask = im
    cv2.imshow("Mask", mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Mask to Image", masked)
    cv2.waitKey(0)
    ee = 0
