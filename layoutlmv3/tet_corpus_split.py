# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/10/9 22:42
# @author  : Mo
# @function: 构建固定的训练-验证数据集


from typing import List, Union, Dict, Any
from collections import Counter
import logging
import random
import pickle
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
print(path_root)

CUDA_VISIBLE_DEVICES = "0"
os.environ["USE_TORCH"] = "1"

from datasets import (Array2D, Array3D, ClassLabel, Dataset, Features, Sequence, Value)
from transformers import AdamW
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


def txt_write(lines: List[str], path: str, model: str = "w", encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logging.info(str(e))


def txt_read(path: str, encoding: str = "utf-8"):
    """
    Read Line of list form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        dict of word2vec, eg. {"macadam":[...]}
    """

    lines = []
    try:
        file = open(path, "r", encoding=encoding)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logging.info(str(e))
    finally:
        return lines


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
    path_files = list(set(path_files))
    path_files.sort()
    return path_files


def get_current_dirs(path_dir):
    """
        递归获取当前的所有目录
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_dirs = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for dir in dirs:  # 遍历文件
            file_path = os.path.join(root, dir)  # 获取文件绝对路径
            path_dirs.append(file_path)  # 将文件路径添加进列表
        break
    path_dirs = list(set(path_dirs))
    path_dirs.sort()  # 保证每次一样
    return path_dirs


corpus_type = "document_classification_3"
path_dataset_dir = os.path.join(path_root, "dataset", corpus_type)
path_dataset_train = os.path.join(path_dataset_dir, "train.json")
path_dataset_dev = os.path.join(path_dataset_dir, "dev.json")
train_rate = 0.8

path_dataset_label_dir = get_current_dirs(path_dataset_dir)
data_train = []
data_dev = []
for p in path_dataset_label_dir:
    label = os.path.split(p)[-1]
    files = get_all_dirs_files(p)
    files = [os.path.join(label, os.path.split(file)[-1])+"\n" for file in files]

    lines = [json.dumps({"label": label, "path_img": s}, ensure_ascii=False) + "\n" for s in files]
    len_rate = int(len(lines) * train_rate)
    data_train.extend(lines[:len_rate])
    data_dev.extend(lines[len_rate:])

random.shuffle(data_train)
txt_write(data_train, path_dataset_train)
txt_write(data_dev, path_dataset_dev)



"""
构建固定的训练-验证数据集
"""

