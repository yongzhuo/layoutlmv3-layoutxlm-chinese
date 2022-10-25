# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/10/01 21:11
# @author  : Mo
# @function: XLMRobertaTokenizer


from transformers import XLMRobertaTokenizer


path_tokenizer = "unilm/layoutlmv3-base-chinese"
tokenizer = XLMRobertaTokenizer.from_pretrained(path_tokenizer)
text = ["hello", "world"]
encoding = tokenizer(text, is_split_into_words=True, max_length=512)
print(encoding)

