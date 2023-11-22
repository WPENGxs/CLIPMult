from label2text import label2text
import argparse

dataset = ['topic', 'emotion', 'situation', 'agnews', 'snips', 'trec', 'subj']
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=int, default=1)
parser.add_argument('--num_of_word', type=int, default=30)
parser.add_argument('--topk', type=int, default=1)

def main():
    args = parser.parse_args()
    print('############################## Match-CLIPMulti start ##############################')
    try:
        m = label2text(args.prompt, args.num_of_word, args.topk)
        for d in dataset:
            print('dataset:', d)
            m.label2text_image(d)
            print('------------------------------------------------------------')
        print('############################## Match-CLIPMulti done. ##############################')
    except:
        print('You have entered the wrong parameter, please check.')
        print('############################## Match-CLIPMulti error ##############################')
    

if __name__ == '__main__':
    main()