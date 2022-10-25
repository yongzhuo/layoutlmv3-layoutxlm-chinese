# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/8/19 16:52
# @author  : Mo
# @function: train model of layoutxlm


from collections import Counter
import logging
import pickle
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
print(path_root)

# CUDA_VISIBLE_DEVICES = "0"
os.environ["USE_TORCH"] = "1"

from datasets import (Array2D, Array3D, ClassLabel, Dataset, Features, Sequence, Value)
from transformers import AdamW, get_linear_schedule_with_warmup
# from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from layoutxlm.model.modeling_layoutlmv2 import LayoutLMv2ForSequenceClassification
from layoutxlm.model.tokenization_layoutxlm_fast import LayoutXLMTokenizerFast
from layoutxlm.model.configuration_extra import LAY_CUDA_VISIBLE_DEVICES
from layoutxlm.model.preprocess_ocr import apply_ocr


## 超参数配置
corpus_type = "document_classification_3"
path_dataset_dir = os.path.join(path_root, "dataset", corpus_type)
model_name_or_path = "microsoft/layoutxlm-base"


path_train_json = os.path.join(path_dataset_dir, "train.json")
path_dev_json = os.path.join(path_dataset_dir, "dev.json")

path_model_save_dir = "output/{}".format(corpus_type)
if not os.path.exists(path_model_save_dir):
    os.makedirs(path_model_save_dir)
path_label2id_id2label = os.path.join(path_model_save_dir, "label2id_id2label.json")
path_model_abs = os.path.join(path_model_save_dir, "last_model.bin")

max_len = 256
batch_size = 8
input_size = 224
ch = 3
test_size = 0.2
epochs = 20
# lr = 1e-5  # 72%, 1epoch, 2error
# lr = 2e-5  # 90%, 1epoch, 2error
lr = 3e-5  # 100%, 1epoch, 1error
# lr = 5e-5  # 96%, 2epoch, 1error
# lr = 8e-5  #
# lr = 8e-4
# lr = 5e-4  #
# lr = 1e-4  # 72%, 0epoch,
# lr = 1e-3  # 66%
# lr = 2e-3 # 60%
# lr = 5e-3
# lr = 1e-2 # 66%
use_cache = True  # 图像预处理--->图像/ocr就中间结果存储与否（更改就删除）
use_toy = True  # True  # 只使用batch*2个数据集
use_limit = False  # 限制len_limit的数据进行训练/验证
pre_save_step = 320
len_limit = 3200
weight_decay = 0.001
num_warmup_steps = 0
freeze_last_layer = None  # -2 # 冻结[0-(-2)], None不冻结


device = torch.device("cuda:{}".format(LAY_CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() and LAY_CUDA_VISIBLE_DEVICES != "-1" else "cpu")


def save_json(lines, path: str, encoding: str = "utf-8", indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()


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


def save_pickle(data, path):
    """ 存储 """
    with open(path, "wb") as fp:
        pickle.dump(data, fp, protocol=4)


def load_pickle(path):
    """ 加载 """
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data


def preprocess_data_json(data_json):
    """  从文件中读取数据集  """
    images = []
    labels = []
    for d in tqdm(data_json, desc="desc"):
        d_json = json.loads(d.strip())
        # {"label": "email", "path_img": "email\\doc_000550.png\n"}
        path_img = os.path.join(path_dataset_dir, d_json.get("path_img", "").strip().replace("\\", "/"))
        # path_img = path_dataset_dir + "/" + d_json.get("path_img", "").strip()
        label = d_json.get("label", "")
        images.append(path_img)
        labels.append(label)
    return images, labels


def image_is_exist(dataset):
    """  检测文本是否存在  """
    images, labels = dataset
    images_new, labels_new = [], []
    for idx, image in enumerate(images):
        try:
            _ = Image.open(image).convert("RGB")
            images_new.append(image)
            labels_new.append(labels[idx])
        except Exception as e:
            print(e)
    return images_new, labels_new


def processor(img_or_path, img_size=224, max_len=512, padding="max_length", truncation=True):
    """ 数据预处理模块(图片 + ocr文本) """
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


def preprocess_data(examples):
    """ 数据预处理训练/验证数据, image图片读取与ocr识别文本 """
    images = examples[0]
    labels = examples[1]

    # print("examples")
    # print(examples)
    # images = examples["image_path"].tolist()
    # labels = examples["label"].tolist()
    # print("images")
    # print(images)
    encoded_inputs_total = {}
    for idx, i in tqdm(enumerate(range(len(images))), desc="---preprocess---"):
        image = images[i]
        encoded_inputs = processor(image, padding="max_length", truncation=True, img_size=input_size)
        encoded_inputs["labels"] = label2id[labels[idx]]
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
                                # encoded_inputs_total["token_type_ids"],
                                encoded_inputs_total["bbox"], encoded_inputs_total["labels"])
    return tensor_data


data_train_json = txt_read(path_train_json)
data_dev_json = txt_read(path_dev_json)

dataset_train = preprocess_data_json(data_train_json)
dataset_dev = preprocess_data_json(data_dev_json)

train_label_counter = Counter(dataset_train[1])
dev_label_counter = Counter(dataset_dev[1])
print("label_counter: ")
print(train_label_counter)
print(dev_label_counter)

labels = [label for label in list(set(dataset_train[1]))]
id2label = {str(v): k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
print(label2id)


if use_toy:
    dataset_train = dataset_train[0][:batch_size*2], dataset_train[1][:batch_size*2]
    dataset_dev = dataset_dev[0][:batch_size*2], dataset_dev[1][:batch_size*2]

if use_limit:
    dataset_train = dataset_train[0][:len_limit], dataset_train[1][:len_limit]
    dataset_dev = dataset_dev[0][:len_limit], dataset_dev[1][:len_limit]


print("训练集: ")
print(len(dataset_train[1]))
print("验证集: ")
print(len(dataset_dev[1]))


tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_name_or_path)

## 缓存数据与否
path_pickle_train = "train.pickle"
path_pickle_dev = "dev.pickle"
if os.path.exists(path_pickle_train) and use_cache:
    print("use pickle")
    train_encoded_data = load_pickle(path_pickle_train)
    label2id_id2label_dict = load_json(path_label2id_id2label)
    label2id = label2id_id2label_dict.get("label2id", {})
    id2label = label2id_id2label_dict.get("id2label", {})
else:
    print("use paddleocr")
    train_encoded_data = preprocess_data(dataset_train)
    save_pickle(train_encoded_data, path_pickle_train)
    label2id_id2label_dict = {"label2id": label2id, "id2label": id2label}
    save_json(label2id_id2label_dict, path_label2id_id2label)

if os.path.exists(path_pickle_dev) and use_cache:
    test_encoded_data = load_pickle(path_pickle_dev)
else:
    test_encoded_data = preprocess_data(dataset_dev)
    save_pickle(test_encoded_data, path_pickle_dev)

print("train_encoded_data:")
print(len(train_encoded_data))
# data loaders
train_dataloader = torch.utils.data.DataLoader(train_encoded_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_encoded_data, batch_size=batch_size)
features = Features({
    'image': Array3D(dtype="int64", shape=(ch, input_size, input_size)),
    'input_ids': Sequence(feature=Value(dtype="int64")),
    'attention_mask': Sequence(Value(dtype="int64")),
    'token_type_ids': Sequence(Value(dtype="int64")),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2id), names=[id2label[str(i)] for i in range(len(id2label))]),
})
model = LayoutLMv2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(id2label))
model.to(device)


def train(model, dataloader, optimizer, epoch_cur):
    """ 训练 """
    pbar = tqdm(dataloader)
    correct = 0
    total_loss = 0
    max_accuracy = 0
    progress = 0
    total_step = 0
    for batch_idx, batch in enumerate(pbar):
        total_step += 1
        # print(batch)
        # label_true = batch[5]
        batch = [b.to(device) for b in batch]
        inputs = {
                  'image': batch[0],
                  'input_ids': batch[1],
                  'attention_mask': batch[2],
                  # 'token_type_ids': batch[3],
                  'bbox': batch[3],
                  # 'return_dict': True,
                  'labels': batch[4]
                  }
        model.train()
        optimizer.zero_grad()
        outputs = model(**inputs)
        # print(outputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        # optimizer.zero_grad()

        predictions = outputs.logits.argmax(-1)
        correct += (predictions == inputs['labels']).float().sum()
        # correct += (predictions == inputs).float().sum()
        total_loss += loss.item()
        # progress += batch["input_ids"].shape[0]
        progress += inputs["input_ids"].shape[0]

        pbar.set_description(desc=f'batch_id={batch_idx} loss={total_loss / (batch_idx+1):.4f} acc={100 * correct / progress:.2f} %')

        if total_step > 0 and total_step % pre_save_step == 0:
            accuracy = evaluate(model, test_dataloader)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                print("acc={}, save-model".format(accuracy))
                torch.save(model.state_dict(), os.path.join(path_model_save_dir, "pytorch_model_{}.bin".format(epoch_cur)))


def evaluate(model, dataloader):
    """ 验证 """
    model.eval()
    total_loss = 0
    correct = 0
    iteration = 0
    with torch.no_grad():
        for batch in dataloader:
            # forward pass
            with torch.no_grad():
                batch = [b.to(device) for b in batch]
                inputs = {
                          'image': batch[0],
                          'input_ids': batch[1],
                          'attention_mask': batch[2],
                          # 'token_type_ids': batch[3],
                          'bbox': batch[3],
                          # 'return_dict': True,
                          'labels': batch[4]
                          }
                outputs = model(**inputs)
                loss = outputs.loss

                total_loss += loss.item()
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == inputs['labels']).float().sum()
                iteration += 1

    accuracy = 100 * correct / len(dataloader.dataset)
    print("loss: {:.4f} \t Accuracy: {:.2f} %\n".format(total_loss / iteration, accuracy.item()))
    return accuracy


### 冻结-froze
if freeze_last_layer is not None:
    for params in list(model.parameters())[:freeze_last_layer]:
        params.requires_grad = False

### bias不L2正则化
params_no_decay = ["LayerNorm.weight", "bias"]
parameters_no_decay = [
    {"params": [p for n, p in model.named_parameters() if not any(pnd in n for pnd in params_no_decay)],
     "weight_decay": weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(pnd in n for pnd in params_no_decay)],
     "weight_decay": 0.0}
    ]
num_training_steps = int(epochs * len(train_dataloader)/batch_size)
optimizer = AdamW(parameters_no_decay, lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

### 训练-验证模型
acc_global = 0
for epoch_cur in range(epochs):
    print("Epoch: ",  epoch_cur)
    train(model, train_dataloader, optimizer, epoch_cur)
    acc_current = evaluate(model, test_dataloader)
    if acc_current > acc_global:
        acc_global = acc_current
        pytorch_model_name = "model_{}_{}.bin".format(epoch_cur, round(float(acc_current.cpu().numpy()), 4))
        path_model_abs_current = os.path.join(os.path.split(path_model_abs)[0], pytorch_model_name)
        torch.save(model.state_dict(), path_model_abs_current)
        torch.save(model.state_dict(), path_model_abs)

# torch.save(model.state_dict(), path_model_abs)


"""
注意, 训练模型显存占用较高, 可减小batch-size/或者是冻结embedding层

# shell
# nohup python tet_train.py > doc.train.layout.log 2>&1 &
# tail -n 1000  -f doc.train.layout.log
# |myz|

Epoch:  0
batch_id=8 loss=0.8719 acc=57.58 %: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  4.47it/s]
loss: 0.8104     Accuracy: 63.64 %

Epoch:  1
batch_id=8 loss=0.6974 acc=78.79 %: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:01<00:00,  4.84it/s]
loss: 0.8009     Accuracy: 63.64 %




# lr = 1e-5  # 72%, 1epoch, 2error
# lr = 2e-5  # 90%, 1epoch, 2error
# lr = 3e-5  # 100%, 1epoch, 1error
# lr = 5e-5  # 96%, 2epoch, 1error

"""