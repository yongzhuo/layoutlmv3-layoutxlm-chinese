# 中(英)文 layoutlmv3 实验
## 一、summary
 - 模型比较敏感, 不怎么稳定;
 - 与BERT-base等相比, 相当于新增image-embedding, bbox的四个位置embedding

## 二、layoutlmv3-embedding-简单使用
```bash

  python tet_embedding.py
  
```

## 三、layoutlmv3文档分类-简单使用
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
 
