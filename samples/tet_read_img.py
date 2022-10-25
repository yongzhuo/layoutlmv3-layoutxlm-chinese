# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/10/13 15:53
# @author  : Mo
# @function:

import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)

import numpy as np
import cv2


corpus_type = "document_classification_3"
path_dataset_dir = os.path.join(path_root, "dataset", corpus_type)
img_or_path = os.path.join(path_dataset_dir, "email", "doc_000042.png")
# img_or_path = "doc_000507.png"

img_or_path = img_or_path.replace("\\", "/")
print(img_or_path)
img = cv2.imdecode(np.fromfile(img_or_path, dtype=np.uint8), cv2.IMREAD_COLOR)
ee = 0

