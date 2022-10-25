# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2022/6/16 21:58
# @author   :Mo
# @function :切图 和 遮挡图


from collections import Counter
import json
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)

from tqdm import tqdm
import numpy as np
import cv2


__all__ = [ "mask_image_from_polygon",
            "crop_image_from_polygon",
            "mask_image_from_bbox",
            "crop_image_from_bbox",
            "fill_image_with_border",
            "sort_bboxs",
           ]


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
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
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


def mask_image_from_polygon(img_or_path, img_coords, img_idx=0, img_class="mask", path_img_out=None, is_draw=False, mask_color=255):
    """
        从多边形polygon切图, crop image from polygon
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        img_coords[List<Tuple>]: coords of img need mask, multi-coordinate, eg. (0.0, 0.0, 255.0, 255.0)
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
    Returns:
        lines[List]: output lines
    """
    flag = False
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
    else:
        flag = True
        img = img_or_path
    for img_coord in img_coords:
        img_coords_np = np.array(img_coord)
        img_coords_rect = cv2.boundingRect(img_coords_np)
        x, y, w, h = img_coords_rect
        img[y:y+h, x:x+w] = mask_color
    if is_draw:
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img)  # 截取的图片
    return img


def mask_image_from_bbox(img_or_path, img_coords, img_idx=0, img_class="mask", path_img_out=None, is_draw=False, mask_color=255):
    """
        从bbox遮挡多个图, mask image
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        img_coords[List(Tuple)]: coord of img, four-coordinate, eg. (0.0, 0.0, 255.0, 255.0)
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
        mask_color[Int]: index of color of image, 0-255, eg. 215
    Returns:
        lines[List]: masked image
    """
    flag = False
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
    else:
        flag = True
        img = img_or_path
    for img_coord in img_coords:
        x1, y1, x2, y2 = img_coord
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img[y1:y2, x1:x2, ::-1] = mask_color
    if is_draw:
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img)  # 截取的图片
    return img


def mask_image_from_not_bbox(img_or_path, img_coords, img_idx=0, img_class="mask_not", path_img_out=None, is_draw=False):
    """
        非points区域置为白化(多个bbox会使得分辨率降低),
    遮挡掩膜, 全黑图像，bbox区域像素改成白色，最后使用cv2.add叠加image和mask就可以实现图像的遮挡显示。
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        img_coords[List(Tuple)]: coord of img, four-coordinate, eg. (0.0, 0.0, 255.0, 255.0)
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
        mask_color[Int]: index of color of image, 0-255, eg. 215
    Returns:
        lines[List]: masked image
    """
    flag = False
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
    else:
        flag = True
        img = img_or_path
    img_res = None
    for img_coord in img_coords:
        x1, y1, x2, y2 = img_coord
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        img_copy = copy.deepcopy(img)
        # 黑色填充非bbox区域
        mask_black = np.zeros([img_copy.shape[0], img_copy.shape[1]], dtype=np.uint8)
        mask_black[y1:y2, x1:x2] = 255
        img_copy = cv2.add(img_copy, np.zeros_like(img_copy, dtype=np.uint8), mask=mask_black)
        # 白色填充
        img_white = np.ones_like(img_copy, np.uint8) * 255
        img_white = cv2.bitwise_not(img_white, img_white, mask=mask_black)
        img_res_one = img_copy + img_white
        # 多个bbox的叠加
        img_res = img_res_one if img_res is None else img_res + img_res_one
    if is_draw:
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img_res)  # 截取的图片
    return img_res


def crop_image_from_polygon(img_or_path, img_coord, img_idx=0, img_class="crop", path_img_out=None, is_draw=False):
    """
        从多边形polygon切图, crop image from polygon
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        img_coord[Tuple]: coord of img, four-coordinate, eg. (0.0, 0.0, 255.0, 255.0)
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
    Returns:
        lines[List]: output lines
    """

    flag = False
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
    else:
        flag = True
        img = img_or_path
    img_coords_np = np.array(img_coord)
    img_coords_rect = cv2.boundingRect(img_coords_np)
    x, y, w, h = img_coords_rect
    img_crop = img[y:y+h, x:x+w]
    if is_draw and img_crop.size > 0:
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img_crop)  # 截取的图片
    return img_crop


def crop_image_from_bbox(img_or_path, img_coord, img_idx=0, img_class="crop", path_img_out=None, is_draw=False):
    """
        从长方形bbox切图, crop image from bbox
    Args:
        img_or_path[Numpy or String]: img or path of img to read, eg. "cnn_network_VGG_architecture.png"
        img_coord[Tuple]: coord of img, four-coordinate, eg. (0.0, 0.0, 255.0, 255.0)
        img_idx[Int]: index of img-out, eg. 6
        img_class[String]: type of class, eg. "type"
        path_img_out[String]: path of file of read, eg. "cnn_network_VGG_architecture.png"
        is_draw[Boolean]: wether draw middle image or not, eg. True or False
    Returns:
        lines[List]: crop image
    """
    flag = False
    if os.path.exists(img_or_path):
        img = cv2.imread(img_or_path)  # 打开图片
    else:
        flag = True
        img = img_or_path
    x1, y1, x2, y2 = img_coord
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    img_crop = img[y1:y2, x1:x2, ::-1]
    if is_draw and img_crop.size > 0:
        if flag and not path_img_out:
            path_img_out = "{}.{}.jpg".format(img_class, img_idx)
        else:
            path_img_out = path_img_out if path_img_out \
                else img_or_path.replace(".", ".{}.{}.".format(img_class, img_idx))
        cv2.imwrite(path_img_out, img_crop)  # 截取的图片
    return img_crop


def sort_bboxs(bboxs):
    """
        多个框图排序
    Args:
        bboxs[List<List>]: many-bbox, eg. [[1,2,3,4], [1,2,5,6]]
    Returns:
        bboxs_sort[List<List>]
    """
    bboxs_np = np.asarray(bboxs)
    # max_width = np.sum(bboxs_np[::, (0, 2)], axis=1).max()
    # max_height = np.max(bboxs_np[::, 3])
    # nearest = max_height * 1.4
    ind_list = np.lexsort((bboxs_np[:, 0], bboxs_np[:, 1]))
    bboxs_sort = bboxs_np[ind_list].tolist()
    # bboxs_np.sort(key=lambda r: [int(nearest * round(float(r[1]) / nearest)), r[0]])
    # bboxs_sort = bboxs_np.tolist()
    return bboxs_sort


def imread_zh(path_img):
    """读取中文数据
    Args:
        path_img[Str]: image path, eg. "012_123.jpg"
    Returns:
        img[np.array]
    """
    img = cv2.imdecode(np.fromfile(path_img, dtype=np.uint8), -1)
    # _, img = cv2.imencode(".jpg", img)
    # img_bytes = img.tobytes()
    return img


if __name__ == '__main__':
    corpus_type = "image_samples"
    path_dataset_dir = os.path.join(path_root, "dataset", corpus_type)
    path_img = os.path.join(path_dataset_dir, "doc_paper_0.jpg")

    # 多边形 切图 和 遮挡
    # path_img = "6.jpg"
    points = [[[0, 0], [100, 500], [300, 50], [400, 1000]], [[100, 100], [200, 600], [400, 150], [500, 1100]]]
    mask_image_from_polygon(path_img, points, is_draw=True)
    crop_image_from_polygon(path_img, points[0], is_draw=True)


    # 长方形 切图 和 遮挡
    # path_img = "6.jpg"
    points = [[107, 650, 1312, 815], [111, 401, 1285, 654], [98, 1239, 1342, 1585]]
    mask_image_from_bbox(path_img, points, is_draw=True)
    crop_image_from_bbox(path_img, points[0], is_draw=True)


    # 填充边界框
    fill_image_with_border(path_img, std_size=2048, is_draw=True)


    ## 根据大图的bbox + crop后小图文字的bbox后, 当前行的位置信息
    # path_img = "6.jpg"
    points = [107, 650, 1312, 815]
    # {'linetext': 'zhí jiē xiě dé shù', 'box': [52, 6, 179, 33]}
    bbox = [52, 6, 179, 33]
    bbox_real = [points[0] + bbox[0], points[1] + bbox[1], points[0] + bbox[2], points[1] + bbox[3]]
    crop_image_from_bbox(path_img, bbox_real, img_idx=32, is_draw=True)
    ee = 0


    # BBOX框排序
    bboxs = [[1168, 16, 103, 39],
                [1170, 22, 100, 43],
                [1181, 16, 105, 41],
                [23, 6, 551, 46]
                ]
    res = sort_bboxs(bboxs)
    print(res)


    # # 非框图区域置0, 并且压缩
    # path_img = "6.jpg"
    points = [[107, 650, 1312, 815], [111, 401, 1285, 654], [98, 1239, 1342, 1585]]
    img_white = mask_image_from_not_bbox(path_img, [points[0]], is_draw=True)
    height, width, channel = img_white.shape

    input_size = 224
    image = cv2.resize(img_white, (input_size, input_size))
    print("image.size: ")
    print(image.shape)
    print(type(image))
    # h,w,3
    # image = np.array(image)
    print(image.size)
    image = image.transpose(2, 0, 1)
    image = image[::-1, :, :]
    print(image.size)
    ee = 0

