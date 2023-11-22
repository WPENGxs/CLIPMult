from CLIPMulti import CLIPMulti
import argparse


dataset = ['topic', 'emotion', 'situation', 'agnews', 'snips', 'trec', 'subj']
combination = ['NSC', 'NWSC', 'MF']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='snips')
parser.add_argument('--combination', type=str, default='all')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--prompt', type=int, default=0)
parser.add_argument('--num_of_word', type=int, default=30)
parser.add_argument('--topk', type=int, default=1)


def main():
    args = parser.parse_args()
    
    if args.test:
        if args.prompt == 0:
            m = CLIPMulti('test', 'text')
            
        else:
            text_str = ''
            if args.prompt == 1:
               text_str += 'l'
            elif args.prompt == 2:
                text_str += 'll'
            elif args.prompt == 3:
                text_str += 'lll'
            else:
                print('No this prompt, using prompt_1')
                text_str += 'l'
            
            text_str += f'_text_{args.num_of_word}_{args.topk}'
            m = CLIPMulti('test', text_str)
            
        if args.combination == 'all':
            print('############################################################')
            print('all_eval')
            print('dataset:', args.dataset)
            if args.dataset == 'subj':
                m.clip_mix_text_image_eval(args.dataset)
            else:
                m.clip_text_image_eval(args.dataset)
                m.clip_con_text_image_eval(args.dataset)
                m.clip_mix_text_image_eval(args.dataset)
            print('############################################################')
        elif args.combination == 'NSC':
            print('############################################################')
            print('NSC_eval')
            print('dataset:', args.dataset)
            if args.dataset == 'subj':
                print('subj don\'t have this combination')
            else:
                m.clip_text_image_eval(args.dataset)
            print('############################################################')
        elif args.combination == 'NWSC':
            print('############################################################')
            print('NWSC_eval')
            print('dataset:', args.dataset)
            if args.dataset == 'subj':
                print('subj don\'t have this combination')
            else:
                m.clip_con_text_image_eval(args.dataset)
            print('############################################################')
        elif args.combination == 'MF':
            print('############################################################')
            print('MF_eval')
            print('dataset:', args.dataset)
            m.clip_mix_text_image_eval(args.dataset)
            print('############################################################')
            
    else:
        print('Using dev or train pleause write your own code, this code provides an evaluation only.')
        # m = CLIPMulti('train', 'text')
        # m = CLIPMulti('dev', 'text')

if __name__ == '__main__':
    main()