# 文档理解-layoutxlm-layoutlmv3-实验
## 一、总结-summary
 - layoutxlm/layoutlmv3模型比较敏感, 不怎么稳定, 尤其是对lr很敏感, 2e-5至5e-5;
 - layoutxlm/layoutlmv3与BERT-base等相比, 相当于新增image-embedding, bbox的四个位置embedding;
 - 个人感觉比较适配表单理解类任务(xfusd), 不怎么适合目标检测等其他细粒度的任务, 更多的还是偏向于NLP任务, image-embedding聊胜于无;
 - 在自己的一个实际文档分类任务中, bert-base的f1都有95%左右, layoutxlm精调结果才90%左右(还很不稳定), 或许是因为(ocr不一样?);
 - （?）使用yolo系列 + bert自己融合或许还比layoutlm系列效果要好, 尤其是细粒度的文档任务;
 - 源码地址为：[https://github.com/yongzhuo/layoutlmv3-layoutxlm-chinese](https://github.com/yongzhuo/layoutlmv3-layoutxlm-chinese)

## 二、layoutxlm-embedding-简单使用
```bash

  python tet_embedding.py
  
```

## 三、layoutxlm文档分类-简单使用
```bash

划分数据集(已完成): python tet_corpus_split.py
训练: python tet_train.py
预测: python tet_pred.py

纯bert-base对比
训练: python tet_bert_train.py
预测: python tet_bert_pred.py

```


## 四、环境要求与安装
详见README_env.md


## 五、reference
 - [unilm]: [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm)
 - [LayoutLMv2 Document Classification]: [https://www.kaggle.com/code/ritvik1909/layoutlmv2-document-classification](https://www.kaggle.com/code/ritvik1909/layoutlmv2-document-classification)
 - [Document Classification:: LayoutLMV2]: [https://www.kaggle.com/code/anantgupt/document-classification-layoutlmv2](https://www.kaggle.com/code/anantgupt/document-classification-layoutlmv2)
 - [RVL-CDIP + LayoutLMv2 Document Classification]: [https://www.kaggle.com/code/lonelvino/rvl-cdip-layoutlmv2-document-classification](https://www.kaggle.com/code/lonelvino/rvl-cdip-layoutlmv2-document-classification)
 
