# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/8/19 16:52
# @author  : Mo
# @function: predict of layoutxlm


from collections import Counter
import logging
import pickle
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
print(path_root)

# CUDA_VISIBLE_DEVICES = "-1"
os.environ["USE_TORCH"] = "1"

from datasets import (Array2D, Array3D, ClassLabel, Dataset, Features, Sequence, Value)
from transformers import AdamW
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from layoutxlm.model.modeling_layoutlmv2 import LayoutLMv2ForSequenceClassification
from layoutxlm.model.tokenization_layoutxlm_fast import LayoutXLMTokenizerFast
from layoutxlm.model.configuration_extra import LAY_CUDA_VISIBLE_DEVICES
from layoutxlm.model.preprocess_ocr import apply_ocr


device = torch.device("cuda:{}".format(LAY_CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() and LAY_CUDA_VISIBLE_DEVICES != "-1" else "cpu")

corpus_type = "document_classification_3"
path_dataset_dir = os.path.join(path_root, "dataset", corpus_type)
model_name_or_path = "microsoft/layoutxlm-base"

path_model_save_dir = "output/{}".format(corpus_type)
path_label2id_id2label = os.path.join(path_model_save_dir, "label2id_id2label.json")
path_model_abs = os.path.join(path_model_save_dir, "last_model.bin")
tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_name_or_path)

max_len = 256
batch_size = 16
input_size = 224
ch = 3
test_size = 0.2
epochs = 20
lr = 3e-5
use_toy = True  # 只使用batch*2个
use_limit = False
pre_save_step = 320
len_limit = 3200


def processor(img_or_path, img_size=224, max_len=512, padding="max_length", truncation=True):
    """ 数据预处理(图片 + ocr文本) """
    img, texts, bboxs = apply_ocr(img_or_path, img_size, max_len)
    data_output = tokenizer(text=texts, boxes=bboxs, padding=padding, truncation=truncation, max_length=max_len, stride=0)
    encoded_inputs = data_output.data
    encoded_inputs["image"] = img.tolist()
    # height, width, _ = img.shape
    # cv2.resize()
    # # image = img.resize((input_size, input_size), Image.BICUBIC)
    # image = np.array(image)
    # image = image.transpose(2, 0, 1)
    # image = image[::-1, :, :]
    return encoded_inputs


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    # if length is None:
    #     length = max([len(x) for x in inputs])
    #
    # pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    # outputs = []
    # for x in inputs:
    #     x = x[:length]
    #     pad_width[0] = (0, length - len(x))
    #     x = np.pad(x, pad_width, 'constant', constant_values=padding)
    #     outputs.append(x.tolist())
    # return outputs
    if length is None:
        length = min(max([len(x) for x in inputs]), 512)
    outputs = []
    for x in inputs:
        if len(x) >= length:
            x = x[:length]
        else:
            x = x + [padding] * (length-len(x))
        outputs.append(x)
    return outputs


def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


def preprocess_data_pred(images):
    """ 预测输入 """
    # take a batch of images
    # images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    # print("examples")
    # print(examples)
    # images = examples["image_path"].tolist()
    # labels = examples["label"].tolist()
    # print("images")
    # print(images)
    encoded_inputs_total = {}
    for idx, i in tqdm(enumerate(range(len(images))), desc="---preprocess---"):
        image = images[i]
        encoded_inputs = processor(image, img_size=224, max_len=512, padding="max_length", truncation=True)
        for k, v in encoded_inputs.items():
            if k not in encoded_inputs_total:
                encoded_inputs_total[k] = [v]
            else:
                encoded_inputs_total[k].append(v)

        ### input-tensor
        # # add labels
        # encoded_inputs["labels"] = torch.tensor([label2id[labels[idx]]], dtype=torch.int64).contiguous()
        # for k,v in encoded_inputs.items():
        #     encoded_inputs[k] = v.to(device)
        #     # encoded_inputs[k] = torch.tensor(v, dtype=torch.int64).contiguous().to(device)
        # encoded_inputs_total.append(encoded_inputs)
    # print(encoded_inputs_total["attention_mask"])
    # print(encoded_inputs_total["labels"])
    # print(encoded_inputs["token_type_ids"])
    # encoded_inputs_total["token_type_ids"] = sequence_padding(encoded_inputs_total["token_type_ids"])
    encoded_inputs_total["attention_mask"] = sequence_padding(encoded_inputs_total["attention_mask"])
    encoded_inputs_total["input_ids"] = sequence_padding(encoded_inputs_total["input_ids"])
    encoded_inputs_total["bbox"] = sequence_padding(encoded_inputs_total["bbox"], padding=[0, 0, 0, 0])

    for k, v in encoded_inputs_total.items():
        # print(k)
        # print(np.array(v[0]).shape)
        # print(len(v))
        encoded_inputs_total[k] = torch.tensor(v, dtype=torch.int64).contiguous()  # .to(device)
    # return encoded_inputs_total

    from torch.utils.data import TensorDataset
    tensor_data = TensorDataset(encoded_inputs_total["image"], encoded_inputs_total["input_ids"], encoded_inputs_total["attention_mask"],
                                encoded_inputs_total["bbox"])
                                # encoded_inputs_total["token_type_ids"], encoded_inputs_total["bbox"])
    return tensor_data


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
    path_files.sort()  # 保证每次一样
    return path_files


def predict(path_addr):
    """  预测  """
    data_batch = preprocess_data_pred([path_addr])
    pred_dataloader = torch.utils.data.DataLoader(data_batch, batch_size=batch_size)
    model.eval()
    for batch in pred_dataloader:
        with torch.no_grad():
            batch = [b.to(device) for b in batch]
            inputs = {'image': batch[0],
                      'input_ids': batch[1],
                      'attention_mask': batch[2],
                      # 'token_type_ids': batch[3],
                      'bbox': batch[3],
                      }
            outputs = model(**inputs)
            # print(outputs)
            # loss = outputs.loss
            logits = outputs.logits
            logits_argmax = logits.argmax(-1).cpu().data.numpy()
            logits_softmax = torch.softmax(logits, dim=-1).cpu().data.numpy()
            # print(loss)
            # print(logits_argmax)
            # 最大概率, 最大概率的类别, 类别-概率, like [{"label_1":0.8, "label_2":0.2}]
            ys_prob = []
            for i in range(logits_softmax.shape[0]):
                ypi = logits_softmax[i].tolist()
                line = {}
                for idx, prob in enumerate(ypi):
                    line[id2label[str(idx)]] = round(prob, 6)
                ys_prob.append(line)
            # print(ys_prob)
            label_prob_sort = sorted(ys_prob[0].items(), key=lambda x:x[1], reverse=True)
            # print([id2label(str(p)) for p in logits_argmax])
            print(path_addr)
            print(label_prob_sort)


label2id_id2label_dict = load_json(path_label2id_id2label)
label2id = label2id_id2label_dict.get("label2id", {})
id2label = label2id_id2label_dict.get("id2label", {})
model = LayoutLMv2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(id2label))
model.to(device)
if os.path.exists(path_model_abs):
    model.load_state_dict(torch.load(path_model_abs, map_location=device))
model.eval()
print("load layoutxlm ok!")


path_image_samples = os.path.join(path_root, "dataset", "image_samples")
files = get_all_dirs_files(path_image_samples)
print(files)
for file in files:
    predict(file)


while True:
    print("请输入图片地址: ")
    addr = input()
    predict(addr)


"""
模型预测

load layoutxlm ok!
---preprocess---: 1it [00:04,  4.17s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_email_0.jpg
[('email', 0.64276), ('resume', 0.230465), ('paper', 0.126774)]
---preprocess---: 1it [00:04,  4.32s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_email_1.png
[('email', 0.460797), ('paper', 0.299418), ('resume', 0.239785)]
---preprocess---: 1it [00:20, 20.30s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_paper_0.jpg
[('paper', 0.505004), ('resume', 0.412321), ('email', 0.082675)]
---preprocess---: 1it [00:10, 10.23s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_paper_1.png
[('paper', 0.59401), ('email', 0.271114), ('resume', 0.134876)]
---preprocess---: 1it [00:07,  7.10s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_resume_0.jpg
[('resume', 0.587846), ('paper', 0.249011), ('email', 0.163143)]
---preprocess---: 1it [00:05,  5.96s/it]
/home/moyzh/layoutlm-image/dataset/image_samples/doc_resume_1.png
[('resume', 0.460567), ('paper', 0.347107), ('email', 0.192326)]
请输入图片地址:


"""

