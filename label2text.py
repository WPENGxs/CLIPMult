import os
import sys

import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm
import clip
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, f1_score

class label2text():
    def __init__(self, prompt, words_num, topk_num):
        if prompt == 1:
            self.prompt = 'l'
        elif prompt == 2:
            self.prompt = 'll'
        elif prompt == 3:
            self.prompt = 'lll'
        self.words_num = words_num
        self.topk_num =topk_num
        self.label_dict = {'snips': ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'],
                           'subj': ['subjective', 'objective'],
                           'trec': ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM'],
                           'agnews': ['business', 'politics', 'sports', 'technology'], 
                           'topic': ['Business & Finance', 'Computers & Internet', 'Education & Reference', 'Entertainment & Music',
                                     'Family & Relationships', 'Health', 'Politics & Government', 'Science & Mathematics', 'Society & Culture', 'Sports'],
                           'emotion': ['anger', 'disgust', 'fear', 'guilt', 'joy', 'love', 'sadness', 'shame', 'surprise', 'noemo'],
                           'situation': ['crimeviolence', 'evac', 'food', 'infra', 'med', 'regimechange', 'search', 'shelter', 'terrorism', 'utils', 'water', 'out-of-domain']
                          }

    def _get_label_loader(self, dataset):
        if self.prompt == 'l':
            f = open('./data/{}/text/label2text_{}.txt'.format(dataset, self.words_num), 'r')
        elif self.prompt == 'll':
            f = open('./data/{}/text/label2text_{}_2.txt'.format(dataset, self.words_num), 'r')
        elif self.prompt == 'lll':
            f = open('./data/{}/text/label2text_{}_3.txt'.format(dataset, self.words_num), 'r')
        input_text = f.read()
        f.close()
        text_list = input_text.split('\n')
        labels = []
        label_texts = []
        for s in text_list:
            s = s.split('\t')
            if len(s) == 2:
                labels.append(s[0])
                # context, reduced_content = sc(s[1], reduce_ratio = 0.8)
                label_texts.append(s[1])
        return labels, label_texts
        
    def _get_image_loader(self, dataset, preprocess, device):
        # image loading
        paths_to_images = glob.glob('./data/{}/image/*.jpeg'.format(dataset))
        images = []
        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)
        for i in range(label_num):
            for path_to_image in paths_to_images:
                if dataset_label[i] == os.path.basename(path_to_image).replace('-0.jpeg', ''):
                    with open(path_to_image, 'rb') as jpeg:
                        image = preprocess(Image.open(path_to_image)).unsqueeze(0).to(device)
                        images.append(image)
        images = torch.cat((images), 0)
        return paths_to_images, images

    def label2text_image(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)

        labels, label_texts = self._get_label_loader(dataset)
        paths_to_images, images = self._get_image_loader(dataset, preprocess, device)

        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)

        with torch.no_grad():
            image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        w_str = ''
        with tqdm(total=label_num) as pbar:
            for i in range(label_num):
                label_text = label_texts[i].split(', ')
                # print(label_text)
                text_input = torch.cat([clip.tokenize(f"{l}") for l in label_text]).to(device)
                # text_input = clip.tokenize([label_text], truncate=True).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # use https://github.com/openai/CLIP#zero-shot-prediction
                # scale_image_text = 100
                logits_text_image = 100 * image_features @ text_features.T
                # logits_text_image = logits_text_image.T

                similarity = logits_text_image.softmax(dim=-1)
                values, indices = similarity[i].topk(self.topk_num)
                indices = indices.tolist()
                str = ''
                for index in indices:
                    str += label_text[index] + ', '
                str = str[:-2]
                w_str += dataset_label[i] + '\t' + labels[i] + ', ' + str +'\n'
                pbar.update(1)

        f = open('./data/{}/text/{}_text_{}_{}.txt'.format(dataset, self.prompt, self.words_num, self.topk_num), 'w')
        f.write(w_str)
        f.close()
