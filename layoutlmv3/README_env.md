# Layoutlxlm环境准备说明

## 1. 总的安装见 requirements.txt
```bash
    pip install -r requirements.txt

    -f https://download.pytorch.org/whl/torch_stable.html
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

## 2. 安装 detectron2（版本0.3/0.6都可以）
```
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2
```

## 3. 安装 torch 以及其他
```
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip install paddlepaddle==2.3.0
    pip install paddlepaddle-gpu==2.3.1.post111
    pip install opencv-python==4.6.0.66
    pip install paddleOCR==2.5.0.3
```

## 4. 安装 PaddleOCR 2.5.0.3 【需要当前最新 paddle 2.3.0】
```
    版本太低
    ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found (required by /home/myz/anaconda3/envs/myzhuo/lib/python3.7/site-packages/paddle/fluid/core_avx.so)
    
    sudo cp /home/myz/myzhuo/cv/layoutlmft/examples/libstdc++.so.6.0.28 /usr/lib/x86_64-linux-gnu

    sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    sudo move /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.bak
    
    sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### 5.[libprotobuf FATAL google/protobuf/stubs/common.cc:83] 版本冲突（推理阶段, 3.1.0?）
    pip install protobuf==3.19.4


### 6.paddle 与 detectron2 冲突
    将site-packages/paddleocr/目录中的所有
    tools.infer
    改为
    paddleocr.tools.infer


### 7.下载、加载数据报错
    可在本地下载，然后复制到远程linux上边，注意, transoformer下包括一堆datasets/metrics/modules/transformers, 最后这个是模型

    TypeError: 'Dataset' object does not support item assignment
    报错来自run_xfun_ser.py, 不支持相加 datasets["train"][c] = datasets["train"][c] + datasets['validation'][c]
    pip install datasets==1.8.0


### 8.tensorboardX报错
    TypeError: __init__() got an unexpected keyword argument 'serialized_options'
    因为PaddleOCR需要protobuf低版本, 如3.10.0, 但是transformers需要protobuf至少3.19.0
    pip install tensorboardX==1.8
    pip install protobuf==3.19.4


