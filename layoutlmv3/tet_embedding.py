# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/8/19 22:52
# @author  : Mo
# @function: sample of layoutlmv3


from collections import Counter
import logging
import pickle
import random
import json
import copy
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
print(path_root)

CUDA_VISIBLE_DEVICES = "-1"
os.environ["USE_TORCH"] = "1"

import numpy as np
import torch

from layoutlmv3.model.modeling_layoutlmv3 import LayoutLMv3ForSequenceClassification, LayoutLMv3Model
from transformers import XLMRobertaTokenizerFast as LayoutLMv3Tokenizer


model_name_or_path = "E:/DATA/bert-model/00_pytorch/layoutlmv3-base-chinese"
# model_name_or_path = "microsoft/layoutlmv3-base-chinese"
device = "cuda:{}".format(CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() and CUDA_VISIBLE_DEVICES != "-1" else "cpu"


__all__ = ["layoutlmv3_extract_embedding",
          "layoutlmv3_multimodel_classification"
          ]


def layoutlmv3_extract_embedding():
    """  抽取特征 """
    seq_len = 224
    images = torch.tensor([np.ones((3, 224, 224)).tolist()], dtype=torch.float32).contiguous()
    input_ids = torch.tensor([[i for i in range(seq_len)]], dtype=torch.int64).contiguous()
    attention_mask = torch.tensor([[1] * seq_len], dtype=torch.int64).contiguous()
    token_type_ids = torch.tensor([[0] * seq_len], dtype=torch.int64).contiguous()
    bbox = torch.tensor([[[1, 2, 3, 4]]*seq_len], dtype=torch.int64).contiguous()

    input_x = {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "bbox": bbox,
    }

    tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)
    model = LayoutLMv3Model.from_pretrained(model_name_or_path)
    model.to(device)

    for name, par in model.named_parameters():
        print(name, "   ", par.size())

    y = model(**input_x)

    print(y)


def layoutlmv3_multimodel_classification():
    """  分类 """
    seq_len = 224
    images = torch.tensor([np.ones((3, 224, 224)).tolist()], dtype=torch.float32).contiguous()
    input_ids = torch.tensor([[i for i in range(seq_len)]], dtype=torch.int64).contiguous()
    attention_mask = torch.tensor([[1] * seq_len], dtype=torch.int64).contiguous()
    token_type_ids = torch.tensor([[0] * seq_len], dtype=torch.int64).contiguous()
    bbox = torch.tensor([[[1, 2, 3, 4]] * seq_len], dtype=torch.int64).contiguous()
    labels = torch.tensor([[1]], dtype=torch.int64).contiguous()

    input_x = {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "bbox": bbox,
        "labels": labels,
    }

    tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_name_or_path)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    model.to(device)

    for name, par in model.named_parameters():
        print(name, "   ", par.size())

    y = model(**input_x)
    print(y)


if __name__ == "__main__":
    layoutlmv3_extract_embedding()

    layoutlmv3_multimodel_classification()


"""

FRU:
FUNSD: We fine-tune LayoutLMv3 for 1,000 steps with a learning rate of 1e−5 and a batch size of 16 for FUNSD, 
CORD:  5e − 5 and 64 for CORD

DLC:
RVL-CDIP: We fine-tune LayoutLMv3 for 20,000 steps with a batch size of 64 and a learning rate of 2e − 5.

DQA:
We fine-tune LayoutLMv3BASE for 100,000 steps with a batch size of 128, 
a learning rate of 3e − 5, and a warmup ratio of 0.48. 

For LayoutLMv3LARGE, the step size, batch size, learning rate and warmup ratio 
are 200,000, 32, 1e−5, and 0.1, respectively


last_hidden_state=sequence_output,
hidden_states=encoder_outputs.hidden_states,
attentions=encoder_outputs.attentions,

"""


