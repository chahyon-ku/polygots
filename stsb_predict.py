import argparse
import os
import numpy as np
import sentence_transformers
import torch


def main():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', type=str, default='sentence-transformers/stsb-xlm-r-multilingual')
    # parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--device', type=str, default='cuda:0', choices=('cpu', 'cuda:0', 'cuda:1'))

    # data
    parser.add_argument('--data_dir', type=str, default='data/augment_only')

    # output
    parser.add_argument('--output_dir', type=str, default='stsb')
    
    
    args = parser.parse_args()

    # model
    model = sentence_transformers.SentenceTransformer(args.model_name, device=args.device)
    
    # data
    for dir_entry in os.scandir(args.data_dir):
        if dir_entry.name.endswith('.conll'):
            print(f'Processing {dir_entry.path}')
            sentences = []
            with open(dir_entry.path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                curr_list = []
                for line in lines:
                    if line.startswith('#'):
                        pass
                    elif len(line) == 1:
                        if len(curr_list):
                            sentence = ' '.join(curr_list)
                            sentences.append(sentence)
                        curr_list = []
                    else:
                        curr_list.append(line.split(' ')[0])
            
            print(f'{len(sentences)} sentences')

            with torch.no_grad():
                embeddings = model.encode(sentences)
        
            # output
            os.makedirs(args.output_dir, exist_ok=True)
            np.savetxt(os.path.join(args.output_dir, dir_entry.name.split('.')[0]), embeddings)
        elif dir_entry.is_dir():
            for dir_entry2 in os.scandir(dir_entry.path):
                if dir_entry2.name.endswith('.conll'):
                    print(f'Processing {dir_entry2.path}')
                    sentences = []
                    with open(dir_entry2.path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                        curr_list = []
                        for line in lines:
                            if line.startswith('#'):
                                pass
                            elif len(line) == 1:
                                if len(curr_list):
                                    sentence = ' '.join(curr_list)
                                    sentences.append(sentence)
                                curr_list = []
                            else:
                                curr_list.append(line.split(' ')[0])
                    
                    print(f'{len(sentences)} sentences')

                    with torch.no_grad():
                        embeddings = model.encode(sentences)
                
                    # output
                    os.makedirs(os.path.join(args.output_dir, dir_entry.name), exist_ok=True)
                    np.savetxt(os.path.join(args.output_dir, dir_entry.name, dir_entry2.name.split('.')[0] + '.txt'), embeddings)


if __name__ == '__main__':
    main()
