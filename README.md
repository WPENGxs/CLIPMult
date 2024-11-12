<!--
 * @Author: Peng Wang
 * 
-->
# CLIPMulti: Explore the performance of multimodal enhanced CLIP for zero-shot text classification

<img src="img/pytorch.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the PyTorch implementation and the data of the paper: *"CLIPMulti: Explore the performance of multimodal enhanced CLIP for zero-shot text classification,"*
Peng Wang, Dagang Li, Xuesi Hu, Yongmei Wang*, Youhua Zhang, ***Computer Speech & Language***. [[paper](https://doi.org/10.1016/j.csl.2024.101748)].

## Current progress  

- [x] This work is accepted by *Computer Speech & Language*.
- [x] Submit paper.
- [x] Prepare code and push.

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

## Cite
<pre>
@article{WANG2025101748,
    title = {CLIPMulti: Explore the performance of multimodal enhanced CLIP for zero-shot text classification},
    journal = {Computer Speech & Language},
    volume = {90},
    pages = {101748},
    year = {2025},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2024.101748},
    url = {https://www.sciencedirect.com/science/article/pii/S0885230824001311},
    author = {Peng Wang and Dagang Li and Xuesi Hu and Yongmei Wang and Youhua Zhang},
    keywords = {Zero-shot text classification, CLIP, Multimodality},
    abstract = {Zero-shot text classification does not require large amounts of labeled data and is designed to handle text classification tasks that lack annotated training data. Existing zero-shot text classification uses either a text–text matching paradigm or a text–image matching paradigm, which shows good performance on different benchmark datasets. However, the existing classification paradigms only consider a single modality for text matching, and little attention is paid to the help of multimodality for text classification. In order to incorporate multimodality into zero-shot text classification, we propose a multimodal enhanced CLIP framework (CLIPMulti), which employs a text–image&text matching paradigm to enhance the effectiveness of zero-shot text classification. Three different image and text combinations are tested for their effects on zero-shot text classification, and a matching method (Match-CLIPMulti) is further proposed to find the corresponding text based on the classified images automatically. We conducted experiments on seven publicly available zero-shot text classification datasets and achieved competitive performance. In addition, we analyzed the effect of different parameters on the Match-CLIPMulti experiments. We hope this work will bring more thoughts and explorations on multimodal fusion in language tasks.}
}
</pre>
