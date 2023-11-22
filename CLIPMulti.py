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

# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from IPython.display import Audio, display
# from model import AudioCLIP
# from utils.transforms import ToTensor1D

# from selective_context import SelectiveContext

class CLIPMulti():
    # eval_mode = ['train', 'dev', 'test']
    # text = ['text', 'l_text']
    def __init__(self, eval_mode, text):
        self.eval_mode = eval_mode
        self.emotion_threshold = 0.25
        self.situation_image_threshold = 0.225
        self.situation_text_threshold = 0.725
        # self.situation_text_threshold = 3 * self.situation_image_threshold
        self.text = text
        self.label_dict = {'snips': ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'],
                           'subj': ['subjective', 'objective'],
                           'trec': ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM'],
                           'agnews': ['business', 'politics', 'sports', 'technology'], 
                           'topic': ['Business & Finance', 'Computers & Internet', 'Education & Reference', 'Entertainment & Music',
                                     'Family & Relationships', 'Health', 'Politics & Government', 'Science & Mathematics', 'Society & Culture', 'Sports'],
                           'emotion': ['anger', 'disgust', 'fear', 'guilt', 'joy', 'love', 'sadness', 'shame', 'surprise', 'noemo'],
                           'situation': ['crimeviolence', 'evac', 'food', 'infra', 'med', 'regimechange', 'search', 'shelter', 'terrorism', 'utils', 'water', 'out-of-domain']
                          }
    
    # dataset = ['emotion', 'situation', 'topic', 'agnews', 'snips', 'trec', 'subj']
    # mode = ['train', 'dev', 'test']
    def _get_dataloader(self, dataset, mode):
        # sc = SelectiveContext(model_type='gpt2', lang='en')
        # read test and initialization
        if dataset == 'emotion' or dataset == 'situation':
            self.eval = 'weighted_f1'
            f = open('./dataset/{}/{}.txt'.format(dataset, mode), 'r')
            input_text = f.read()
            f.close()
            text_list = input_text.split('\n')
            labels = []
            sentences = []
            for s in text_list:
                s = s.split('\t')
                if len(s) == 2:
                    mult_s = ''
                    for ss in s[0].split(' '):
                        mult_s += ss + '#'
                    labels.append(mult_s[:-1])
                    # context, reduced_content = sc(s[1], reduce_ratio = 0.8)
                    sentences.append(s[1])
        else:
            self.eval = 'acc'
            f = open('./dataset/{}/{}.txt'.format(dataset, mode), 'r')
            input_text = f.read()
            f.close()
            text_list = input_text.split('\n')
            labels = []
            sentences = []
            for s in text_list:
                s = s.split('\t')
                if len(s) == 2:
                    labels.append(s[0])
                    # context, reduced_content = sc(s[1], reduce_ratio = 0.8)
                    sentences.append(s[1])
        return labels, sentences
    
    def _get_text_loader(self, dataset):
        # text loading
        text_labels = []
        text_sentences = []

        t = open('./data/{}/text/{}.txt'.format(dataset, self.text), 'r')
        input_text = t.read()
        t.close()
        text_list = input_text.split('\n')
        for s in text_list:
            s = s.split('\t')
            if len(s) == 2:
                text_labels.append(s[0])
                text_sentences.append(s[1])
        return text_labels, text_sentences
    
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

    def mult_eval(self, onehot_preds, onehot_golds):
        return {
            'macro_f1': f1_score(y_true=onehot_golds, y_pred=onehot_preds, average='macro'),
            'micro_f1': f1_score(y_true=onehot_golds, y_pred=onehot_preds, average='micro'),
            'weighted_f1': self.cal_wf1(onehot_preds, onehot_golds),
            'acc': accuracy_score(y_true=onehot_golds, y_pred=onehot_preds),
        }
        
    def cal_wf1(self, onehot_preds, onehot_golds):
        num_labels = len(onehot_golds[0])
        onehot_preds = np.array(onehot_preds, dtype=int)
        onehot_golds = np.array(onehot_golds, dtype=int)
    
        wf1 = 0.0
        tot_weight = 0
        for i in range(num_labels):
            f1 = f1_score(onehot_golds[:, i], onehot_preds[:, i], pos_label=1, average='binary')
            weight = sum(onehot_golds[:, i])
            wf1 += weight * f1
            tot_weight += weight
        wf1 = wf1 / tot_weight
        return wf1
    
    def clip_text_eval(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        labels, sentences = self._get_dataloader(dataset, self.eval_mode)
        text_labels, text_sentences = self._get_text_loader(dataset)
        all_num = len(sentences)
        predict_list = []
        
        # text = [[s] for s in text_sentences]
        text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)

        if dataset == 'emotion':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits = logits_text_sentence.T
                    
                    # delete noemo    
                    logits = logits[:, :-1]

                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    # print(max_probs, preds)
                    preds = torch.where(max_probs >= self.emotion_threshold, preds, label_num - 1)

                    onehot_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    onehot_pred[preds] = 1
                    onehot_gold[dataset_label.index(labels[i])] = 1
                    onehot_preds.append(onehot_pred)
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Text eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        elif dataset == 'situation':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits = logits_text_sentence.T
                    
                    onehot_preds_with_ood = torch.zeros(logits.shape)
                    logits = logits[:, :-1]
                    onehot_pred = torch.where(logits >= 100 * self.situation_text_threshold, 1, 0)

                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    label = labels[i].split('#')
                    for ii in range(label_num):
                        for jj in range(len(label)):
                            if dataset_label[ii] == label[jj]:
                                onehot_gold[ii] = 1
                    
                    if torch.equal(onehot_pred[0], torch.zeros(onehot_pred[0].shape).to(device)):
                        onehot_preds_with_ood[:, -1] = 1
                    else:
                        onehot_preds_with_ood[:, :-1] = onehot_pred[0]
                        onehot_preds_with_ood[:, -1] = 0
                    onehot_pred = onehot_preds_with_ood
                    onehot_preds.append(onehot_pred[0].tolist())
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Text eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        else:
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    # set scale_text_text = 100
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    # print(logits_text_sentence)
                    # confidence = logits_text_sentence.softmax(dim=0)
                    x = torch.argmax(logits_text_sentence[:], dim=1)
                    # print(confidence)
                    if text_labels[x.item()] == labels[i]:
                        predict_list.append(1)
                    else:
                        predict_list.append(0)
                    pbar.update(1)
                    
            print("Text Acc", predict_list.count(1)/all_num)
        
    def clip_image_eval(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        labels, sentences = self._get_dataloader(dataset, self.eval_mode)
        paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
        all_num = len(sentences)
        predict_list = []

        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)

        with torch.no_grad():
            image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if dataset == 'emotion':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits = logits_image_sentence.T
                    
                    # logits = logits.tolist()
                    # # delete noemo    
                    # label_sim = []
                    # for ii in range(label_num):
                    #     for jj in range(label_num):
                    #         if dataset_label[ii] == os.path.basename(paths_to_images[jj]).replace('-0.jpeg', ''):
                    #             label_sim.append(logits[0][jj])
                    # label_sim = torch.tensor([label_sim])
                    # delete noemo
                    logits = logits[:, :-1]

                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    # print(max_probs, preds)
                    preds = torch.where(max_probs >= self.emotion_threshold, preds, label_num - 1)

                    onehot_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    onehot_pred[preds] = 1
                    onehot_gold[dataset_label.index(labels[i])] = 1
                    onehot_preds.append(onehot_pred)
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        elif dataset == 'situation':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits = logits_image_sentence.T
                    
                    onehot_preds_with_ood = torch.zeros(logits.shape)
                    logits = logits[:, :-1]
                    onehot_pred = torch.where(logits >= 100 * self.situation_image_threshold, 1, 0)

                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    label = labels[i].split('#')
                    for ii in range(label_num):
                        for jj in range(len(label)):
                            if dataset_label[ii] == label[jj]:
                                onehot_gold[ii] = 1
                    
                    if torch.equal(onehot_pred[0], torch.zeros(onehot_pred[0].shape).to(device)):
                        onehot_preds_with_ood[:, -1] = 1
                    else:
                        onehot_preds_with_ood[:, :-1] = onehot_pred[0]
                        onehot_preds_with_ood[:, -1] = 0
                    onehot_pred = onehot_preds_with_ood
                    onehot_preds.append(onehot_pred[0].tolist())
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        else:
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T

                    x = torch.argmax(logits_image_sentence[:], dim=1)
                    if dataset_label[x.item()] == labels[i]:
                        predict_list.append(1)
                    else:
                        predict_list.append(0)
                    pbar.update(1)
        
            print("Image Acc", predict_list.count(1)/all_num)
        
    # 归一化
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    # 标准化
    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    
    def sort_to_num(self, data):
        s_data = sorted(data, key = float)
        for i in range(len(data)):
            data[i] = s_data.index(data[i])
        return data
            
    def clip_text_image_eval(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        labels, sentences = self._get_dataloader(dataset, self.eval_mode)
        text_labels, text_sentences = self._get_text_loader(dataset)
        paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
        all_num = len(sentences)
        predict_list = []

        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)
        label_text_weight = []
        
        text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            image_features = model.encode_image(images)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if dataset == 'emotion':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T

                    logits_text_sentence = logits_text_sentence[:, :-1]
                    
                    # logits_image_sentence = logits_image_sentence.tolist()
                    # # delete noemo    
                    # label_i = []
                    # for ii in range(label_num):
                    #     for jj in range(label_num):
                    #         if dataset_label[ii] == os.path.basename(paths_to_images[jj]).replace('-0.jpeg', ''):
                    #             label_i.append(logits_image_sentence[0][jj])
                    # label_i = torch.tensor([label_i]).to(device)
                    logits_image_sentence = logits_image_sentence[:, :-1]

                    # sim_text = logits_text_sentence.tolist()
                    sim_text = torch.softmax(logits_text_sentence, dim=-1)
                    # sim_image = logits_image_sentence.tolist()
                    sim_image = torch.softmax(logits_image_sentence, dim=-1)
                    label_sim = []
                    for ii in range(label_num - 1):
                        sim = 0
                        sim += sim_text[0][ii]
                        sim += sim_image[0][ii]
                        label_sim.append(sim / 2)
                    probs = torch.tensor(label_sim)
                    # probs = torch.softmax(label_sim, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    # print(max_probs, preds)
                    preds = torch.where(max_probs >= self.emotion_threshold, preds, label_num - 1)

                    onehot_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    onehot_pred[preds] = 1
                    onehot_gold[dataset_label.index(labels[i])] = 1
                    onehot_preds.append(onehot_pred)
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        elif dataset == 'situation':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T

                    onehot_preds_with_ood = torch.zeros(logits_image_sentence.shape)

                    logits_text_sentence = logits_text_sentence[:, :-1]
                    logits_image_sentence = logits_image_sentence[:, :-1]

                    label_sim = []
                    for ii in range(label_num - 1):
                        sim = 0
                        sim += logits_text_sentence[0][ii]
                        sim += logits_image_sentence[0][ii]
                        label_sim.append(sim / 2)
                    logits = torch.tensor(label_sim).float()

                    onehot_pred = torch.where(logits >= 100 * (self.situation_text_threshold + self.situation_image_threshold) / 2, 1, 0)

                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    label = labels[i].split('#')
                    for ii in range(label_num):
                        for jj in range(len(label)):
                            if dataset_label[ii] == label[jj]:
                                onehot_gold[ii] = 1
                    
                    if torch.equal(onehot_pred[0], torch.zeros(onehot_pred[0].shape)):
                        onehot_preds_with_ood[:, -1] = 1
                    else:
                        onehot_preds_with_ood[:, :-1] = onehot_pred[0]
                        onehot_preds_with_ood[:, -1] = 0
                    onehot_pred = onehot_preds_with_ood
                    onehot_preds.append(onehot_pred[0].tolist())
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        else:
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    # set scale_text_text = 100
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    logits_text_sentence = self.normalization(logits_text_sentence[0].tolist())
                    # logits_text_sentence = self.sort_to_num(logits_text_sentence[0].tolist())
                    # logits_text_sentence = self.standardization(logits_text_sentence[0].tolist())
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T
                    logits_image_sentence = self.normalization(logits_image_sentence[0].tolist())
                    # logits_image_sentence = self.sort_to_num(logits_image_sentence[0].tolist())
                    # logits_image_sentence = self.standardization(logits_image_sentence[0].tolist())
                    
                    label_sim = []
                    for ii in range(label_num):
                        sim = 0
                        sim += logits_text_sentence[ii]
                        sim += logits_image_sentence[ii]
                        label_sim.append(sim)
                        
                    ind = label_sim.index(max(label_sim))
                    # print(confidence)
                    if dataset_label[ind] == labels[i]:
                        predict_list.append(1)
                    else:
                        predict_list.append(0)
                    pbar.update(1)
            
            print("Text & Image Acc", predict_list.count(1)/all_num)

    def clip_con_text_image_eval(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        labels, sentences = self._get_dataloader(dataset, self.eval_mode)
        text_labels, text_sentences = self._get_text_loader(dataset)
        paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
        all_num = len(sentences)
        predict_list = []

        # con
        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)
        label_text_weight = []
        for ii in range(label_num):
            for jj in range(label_num):
                if dataset_label[ii] == os.path.basename(paths_to_images[jj]).replace('-0.jpeg', ''):
                    text_inputs = torch.cat([clip.tokenize(text_sentences[ii])]).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_inputs)
                        image_features = model.encode_image(preprocess(Image.open(paths_to_images[jj])).unsqueeze(0).to(device))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits_text_image = 100 * text_features @ image_features.T
                    label_text_weight.append(logits_text_image[0][0] / 100) 
        
        text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            image_features = model.encode_image(images)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if dataset == 'emotion':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T

                    logits_text_sentence = logits_text_sentence[:, :-1]
                    
                    # logits_image_sentence = logits_image_sentence.tolist()
                    # # delete noemo    
                    # label_i = []
                    # for ii in range(label_num):
                    #     for jj in range(label_num):
                    #         if dataset_label[ii] == os.path.basename(paths_to_images[jj]).replace('-0.jpeg', ''):
                    #             label_i.append(logits_image_sentence[0][jj])
                    # label_i = torch.tensor([label_i]).to(device)
                    logits_image_sentence = logits_image_sentence[:, :-1]

                    sim_text = torch.softmax(logits_text_sentence, dim=-1)
                    sim_image = torch.softmax(logits_image_sentence, dim=-1)
                    label_sim = []
                    for ii in range(label_num - 1):
                        sim = 0
                        sim += label_text_weight[ii] * sim_text[0][ii]
                        sim += (1 - label_text_weight[ii]) * sim_image[0][ii]
                        label_sim.append(sim)
                    probs = torch.tensor(label_sim).float()
                    # probs = torch.softmax(label_sim, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    # print(max_probs, preds)
                    preds = torch.where(max_probs >= self.emotion_threshold, preds, label_num - 1)

                    onehot_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    onehot_pred[preds] = 1
                    onehot_gold[dataset_label.index(labels[i])] = 1
                    onehot_preds.append(onehot_pred)
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Con Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        elif dataset == 'situation':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T

                    onehot_preds_with_ood = torch.zeros(logits_image_sentence.shape)

                    logits_text_sentence = logits_text_sentence[:, :-1]
                    logits_image_sentence = logits_image_sentence[:, :-1]

                    label_sim = []
                    for ii in range(label_num - 1):
                        sim = 0
                        sim += label_text_weight[ii] * logits_text_sentence[0][ii]
                        sim += (1 - label_text_weight[ii]) * logits_image_sentence[0][ii]
                        label_sim.append(sim)
                    logits = torch.tensor(label_sim).float()

                    onehot_pred = torch.where(logits >= 100 * (self.situation_text_threshold + self.situation_image_threshold) / 2, 1, 0)

                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    label = labels[i].split('#')
                    for ii in range(label_num):
                        for jj in range(len(label)):
                            if dataset_label[ii] == label[jj]:
                                onehot_gold[ii] = 1
                    
                    if torch.equal(onehot_pred[0], torch.zeros(onehot_pred[0].shape)):
                        onehot_preds_with_ood[:, -1] = 1
                    else:
                        onehot_preds_with_ood[:, :-1] = onehot_pred[0]
                        onehot_preds_with_ood[:, -1] = 0
                    onehot_pred = onehot_preds_with_ood
                    onehot_preds.append(onehot_pred[0].tolist())
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Con Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        else:
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    # set scale_text_text = 100
                    logits_text_sentence = 100 * text_features @ sentence_features.T
                    logits_text_sentence = logits_text_sentence.T
                    logits_text_sentence = self.normalization(logits_text_sentence[0].tolist())
                    # logits_text_sentence = self.sort_to_num(logits_text_sentence[0].tolist())
                    # logits_text_sentence = self.standardization(logits_text_sentence[0].tolist())
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_image_sentence = 100 * image_features @ sentence_features.T
                    logits_image_sentence = logits_image_sentence.T
                    logits_image_sentence = self.normalization(logits_image_sentence[0].tolist())
                    # logits_image_sentence = self.sort_to_num(logits_image_sentence[0].tolist())
                    # logits_image_sentence = self.standardization(logits_image_sentence[0].tolist())
                    
                    label_sim = []
                    for ii in range(label_num):
                        sim = 0
                        sim += label_text_weight[ii] * logits_text_sentence[ii]
                        sim += (1 - label_text_weight[ii]) * logits_image_sentence[ii]
                        label_sim.append(sim)
                        
                    ind = label_sim.index(max(label_sim))
                    # print(confidence)
                    if dataset_label[ind] == labels[i]:
                        predict_list.append(1)
                    else:
                        predict_list.append(0)
                    pbar.update(1)
            
            print("Con Text & Image Acc", predict_list.count(1)/all_num)
            
    def clip_mix_text_image_eval(self, dataset):
        print('load model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        labels, sentences = self._get_dataloader(dataset, self.eval_mode)
        text_labels, text_sentences = self._get_text_loader(dataset)
        paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
        all_num = len(sentences)
        predict_list = []

        dataset_label = self.label_dict[f'{dataset}']
        label_num = len(dataset_label)
        label_text_weight = []
        
        text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            image_features = model.encode_image(images)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        mix_features = (text_features + image_features) / 2
        
        if dataset == 'emotion':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

                    logits = 100 * mix_features @ sentence_features.T
                    logits = logits.T
                    
                    logits = logits[:, :-1]

                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    # print(max_probs, preds)
                    preds = torch.where(max_probs >= self.emotion_threshold, preds, label_num - 1)

                    onehot_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    onehot_pred[preds] = 1
                    onehot_gold[dataset_label.index(labels[i])] = 1
                    onehot_preds.append(onehot_pred)
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Mix Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        elif dataset == 'situation':
            onehot_preds = []
            onehot_golds = []
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
                    
                    # use https://github.com/openai/CLIP#zero-shot-prediction
                    # scale_image_text = 100
                    logits_mix_sentence = 100 * mix_features @ sentence_features.T
                    logits = logits_mix_sentence.T
                    
                    onehot_preds_with_ood = torch.zeros(logits.shape)
                    logits = logits[:, :-1]
                    onehot_pred = torch.where(logits >= 100 * (self.situation_text_threshold + self.situation_image_threshold) / 2, 1, 0)

                    onehot_gold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    label = labels[i].split('#')
                    for ii in range(label_num):
                        for jj in range(len(label)):
                            if dataset_label[ii] == label[jj]:
                                onehot_gold[ii] = 1
                    
                    if torch.equal(onehot_pred[0], torch.zeros(onehot_pred[0].shape).to(device)):
                        onehot_preds_with_ood[:, -1] = 1
                    else:
                        onehot_preds_with_ood[:, :-1] = onehot_pred[0]
                        onehot_preds_with_ood[:, -1] = 0
                    onehot_pred = onehot_preds_with_ood
                    onehot_preds.append(onehot_pred[0].tolist())
                    onehot_golds.append(onehot_gold)
                    
                    pbar.update(1)
            print('Mix Text & Image eval')
            print(self.mult_eval(onehot_preds, onehot_golds))
        else:
            with tqdm(total=all_num) as pbar:
                for i in range(all_num):
                    sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
                    with torch.no_grad():
                        sentence_features = model.encode_text(sentence_input)
                    sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

                    logits = 100 * mix_features @ sentence_features.T
                    logits = logits.T

                    x = torch.argmax(logits[:], dim=1)
                    # print(confidence)
                    if dataset_label[x.item()] == labels[i]:
                        predict_list.append(1)
                    else:
                        predict_list.append(0)
                    pbar.update(1)
            
            print("Mix Text & Image Acc", predict_list.count(1)/all_num)

    # def clip_mult_eval(self, dataset):
    #     print('load model')
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     # device = 'cpu'
    #     model, preprocess = clip.load("ViT-B/32", device=device)
    #     labels, sentences = self._get_dataloader(dataset, self.eval_mode)
    #     text_labels, text_sentences = self._get_text_loader(dataset)
    #     paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
    #     all_num = len(sentences)
    #     predict_list = []

    #     dataset_label = self.label_dict[f'{dataset}']
    #     label_num = len(dataset_label)
    #     label_text_weight = []
        
    #     text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
    #     with torch.no_grad():
    #         text_features = model.encode_text(text_inputs)
    #         image_features = model.encode_image(images)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
        
    #     mix_features = (text_features + image_features) / 2

    #     with tqdm(total=all_num) as pbar:
    #         for i in range(all_num):
    #             sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
    #             with torch.no_grad():
    #                 sentence_features = model.encode_text(sentence_input)
    #             sentence_features /= sentence_features.norm(dim=-1, keepdim=True)

    #             logits_text_sentence = 100 * text_features @ sentence_features.T
    #             logits_text_sentence = logits_text_sentence.T
    #             logits_text_sentence = self.normalization(logits_text_sentence[0].tolist())
                
    #             logits_image_sentence = 100 * image_features @ sentence_features.T
    #             logits_image_sentence = logits_image_sentence.T
    #             logits_image_sentence = self.normalization(logits_image_sentence[0].tolist())
                
    #             logits_mix_sentence = 100 * mix_features @ sentence_features.T
    #             logits_mix_sentence = logits_mix_sentence.T
    #             logits_mix_sentence = self.normalization(logits_mix_sentence[0].tolist())

    #             label_sim = []
    #             for ii in range(label_num):
    #                 sim = 0
    #                 sim += logits_text_sentence[ii]
    #                 sim += logits_image_sentence[ii]
    #                 sim += logits_mix_sentence[ii]
    #                 label_sim.append(sim)
                    
    #             ind = label_sim.index(max(label_sim))
                
    #             if dataset_label[ind] == labels[i]:
    #                 predict_list.append(1)
    #             else:
    #                 predict_list.append(0)
    #             pbar.update(1)
                
    #     print("Mult Text & Image Acc", predict_list.count(1)/all_num)
    # def _get_audio_loader(self, dataset, aclp, audio_transforms, SAMPLE_RATE):
    #     # audio loading
    #     paths_to_audio = glob.glob('./data/{}/audio/*.wav'.format(dataset))

    #     audio = list()
    #     shape_list = []
    #     for path_to_audio in paths_to_audio:
    #         track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

    #         # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    #         # thus, the actual time-frequency representation will be visualized
    #         # spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    #         # spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    #         # pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

    #         audio.append(track)
    #         shape_list.append(track.shape[0])
    #     # align list
    #     max_shape = max(shape_list)
    #     for i in range(len(audio)):
    #         x = np.zeros((1, max_shape - shape_list[i]))
    #         audio[i] = np.append(audio[i], x[0])
    #     audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track in audio])
        
    #     return paths_to_audio, audio

    # def clip_audio_eval(self, dataset):
    #     print('load model')
    #     torch.set_grad_enabled(False)
    #     MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
    #     # derived from ESResNeXt
    #     SAMPLE_RATE = 44100
    #     # Model Instantiation
    #     aclp = AudioCLIP(pretrained=f'./assets/{MODEL_FILENAME}')
    #     audio_transforms = ToTensor1D()
    #     labels, sentences = self._get_dataloader(dataset, self.eval_mode)
    #     paths_to_audio, audio = self._get_audio_loader(dataset, aclp, audio_transforms, SAMPLE_RATE)
    #     all_num = len(sentences)
    #     predict_list = []
        
    #     ((audio_features, _, _), _), _ = aclp(audio=audio)
    #     audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    #     scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)

    #     dataset_label = self.label_dict[f'{dataset}']
    #     al = []
    #     e = []
    #     with tqdm(total=all_num) as pbar:
    #         for i in range(all_num):
    #             ((_, _, sentence_features), _), _ = aclp(text=[[sentences[i]]])
    #             sentence_features = sentence_features / torch.linalg.norm(sentence_features, dim=-1, keepdim=True)
    #             # set scale_text_text = 100
    #             logits_audio_text = scale_audio_text * audio_features @ sentence_features.T
    #             logits_audio_text = logits_audio_text.T
    #             # print(logits_text_sentence)
    #             # confidence = logits_text_sentence.softmax(dim=0)
    #             x = torch.argmax(logits_audio_text[:], dim=1)
    #             # print(confidence)

    #             if os.path.basename(paths_to_audio[x.item()]).replace('.wav', '') == labels[i]:
    #                 al.append(labels[i])
    #                 predict_list.append(1)
    #             else:
    #                 al.append(labels[i])
    #                 e.append(labels[i])
    #                 predict_list.append(0)
    #             pbar.update(1)
    #     for l in dataset_label:
    #         print(l, e.count(l)/al.count(l))
            
    #     print("Audio Acc", predict_list.count(1)/all_num)

    # def clip_mult_eval(self, dataset):
    #     print('load model')
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     model, preprocess = clip.load("ViT-B/32", device=device)

    #     MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
    #     # derived from ESResNeXt
    #     SAMPLE_RATE = 44100
    #     # Model Instantiation
    #     aclp = AudioCLIP(pretrained=f'./assets/{MODEL_FILENAME}')
    #     audio_transforms = ToTensor1D()
        
    #     labels, sentences = self._get_dataloader(dataset, self.eval_mode)
    #     text_labels, text_sentences = self._get_text_loader(dataset)
    #     paths_to_images, images = self._get_image_loader(dataset, preprocess, device)
    #     paths_to_audio, audio = self._get_audio_loader(dataset, aclp, audio_transforms, SAMPLE_RATE)
    #     all_num = len(sentences)
    #     predict_list = []

    #     ((audio_features, _, _), _), _ = aclp(audio=audio)
    #     audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    #     scale_audio_sentence = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        
    #     # text = [[s] for s in text_sentences]
    #     text_inputs = torch.cat([clip.tokenize(s) for s in text_sentences]).to(device)
    #     with torch.no_grad():
    #         text_features = model.encode_text(text_inputs)
    #         image_features = model.encode_image(images)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)

    #     with tqdm(total=all_num) as pbar:
    #         for i in range(all_num):
    #             sentence_input = clip.tokenize([sentences[i]], truncate=True).to(device)
    #             with torch.no_grad():
    #                 sentence_features = model.encode_text(sentence_input)
    #             sentence_features /= sentence_features.norm(dim=-1, keepdim=True)
    #             # set scale_text_text = 100
    #             logits_text_sentence = 100 * text_features @ sentence_features.T
    #             logits_text_sentence = logits_text_sentence.T
    #             logits_text_sentence = self.normalization(logits_text_sentence[0].tolist())
    #             # logits_text_sentence = self.sort_to_num(logits_text_sentence[0].tolist())
    #             # logits_text_sentence = self.standardization(logits_text_sentence[0].tolist())
                
    #             # use https://github.com/openai/CLIP#zero-shot-prediction
    #             # scale_image_text = 100
    #             logits_image_sentence = 100 * image_features @ sentence_features.T
    #             logits_image_sentence = logits_image_sentence.T
    #             logits_image_sentence = self.normalization(logits_image_sentence[0].tolist())
    #             # logits_image_sentence = self.sort_to_num(logits_image_sentence[0].tolist())
    #             # logits_image_sentence = self.standardization(logits_image_sentence[0].tolist())

    #             ((_, _, sentence_features), _), _ = aclp(text=[[sentences[i]]])
    #             sentence_features = sentence_features / torch.linalg.norm(sentence_features, dim=-1, keepdim=True)
    #             logits_audio_sentence = scale_audio_sentence * audio_features @ sentence_features.T
    #             logits_audio_sentence = logits_audio_sentence.T
    #             logits_audio_sentence = self.normalization(logits_audio_sentence[0].tolist())
                
    #             dataset_label = self.label_dict[f'{dataset}']
    #             label_num = len(dataset_label)
    #             label_sim = []
    #             for ii in range(label_num):
    #                 sim = 0
    #                 for jj in range(label_num):
    #                     if dataset_label[ii] == text_labels[jj]:
    #                         sim += logits_text_sentence[jj]
    #                     if dataset_label[ii] == os.path.basename(paths_to_images[jj]).replace('-0.jpeg', ''):
    #                         sim += logits_image_sentence[jj]
    #                     if dataset_label[ii] == os.path.basename(paths_to_audio[jj]).replace('.wav', ''):
    #                         sim += logits_audio_sentence[jj]
    #                 label_sim.append(sim)
                    
    #             ind = label_sim.index(max(label_sim))
    #             # print(confidence)
    #             if dataset_label[ind] == labels[i]:
    #                 predict_list.append(1)
    #             else:
    #                 predict_list.append(0)
    #             pbar.update(1)
                
    #     print("Text & Image & Aduio Acc", predict_list.count(1)/all_num)