<!--
 * @Author: Peng Wang
 * 
-->
# CLIPMulti

<img src="img/pytorch.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the PyTorch implementation and the data of the paper: **CLIPMulti**.
[AAA](), [BBB](), [CCC](), [DDD](), [EEE]().  ***Computer Speech & Language***.[[paper]]().

## Current progress  

- [x] Submit paper
- [ ] Prepare code and push

## Prerequisites

This codebase was developed and tested with the following settings:

```
scikit-learn==1.2.1
numpy==1.24.2
pytorch==2.0.1
torchvision==0.15.2
tqdm==4.64.1
clip==1.0
transformers==4.27.1
regex==2022.10.31
ftfy==6.1.1
pillow==9.4.0
```

CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```

## How to run it

**CLIPMulti**

run main.py:

```shell
python main.py --dataset [dataset] --combination [combination] --prompt [prompt] --num_of_word [number_of word] --topk [number_of_topk] --test
```

`[dataset]` in `['topic', 'emotion', 'situation', 'agnews', 'snips', 'trec', 'subj']`

`[combination]` in `['NSC', 'NWSC', 'MF']`  
dataset `[subj]` only use `['MF']`

`[prompt]` in `[0, 1, 2, 3]`  
prompt_0 is text written by humans based on labels  
other is Match-CLIPMulti generation

`[number_of word]` in `[30, 40, 50]`  
prompt_1 have `[30, 40, 50]`, prompt_2 and prompt_3 only have `[30]`

`[number_of_topk]` in `[1, 2, 3]`

**Match-CLIPMulti**

run label2text_main.py:

```shell
python label2text_main.py --prompt [prompt] --num_of_word [number_of word] --topk [number_of_topk]
```

`[prompt]` in `[1, 2, 3]`  

`[number_of word]` in `[30, 40, 50]`  
prompt_1 have `[30, 40, 50]`, prompt_2 and prompt_3 only have `[30]`

`[number_of_topk]` in `[1, 2, 3]`

<!-- ## Our paper
<pre>
@inproceedings{qin-etal-2023-cliptext,
    title = "{CLIPT}ext: A New Paradigm for Zero-shot Text Classification",
    author = "Qin, Libo  and
      Wang, Weiyun  and
      Chen, Qiguang  and
      Che, Wanxiang",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.69",
    doi = "10.18653/v1/2023.findings-acl.69",
    pages = "1077--1088",
}
</pre> -->
