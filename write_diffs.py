import argparse
import os
import tqdm


def read_conll(file_name):
    conll = []
    name = ''
    words = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            line = line.strip()
            if line.strip() == '' or line.startswith('# id'):
                if len(words):
                    conll.append({'name': name, 'words': words, 'labels': labels})
                name = line
                words = []
                labels = []
            else:
                word = line.split(' ')[0].strip()
                label = line.split(' ')[3].strip()
                words.append(word)
                labels.append(label)
        
        if len(words):
            conll.append({'name': name, 'words': words, 'labels': labels})
    print('conll built')
    return conll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/MultiCoNER2/en-dev.conll')
    parser.add_argument('--pred_path', type=str, default='preds/last/base/en.conll')
    parser.add_argument('--output_path', type=str, default='preds/last/base/en.diff')
    args = parser.parse_args()

    data = read_conll(args.data_path)
    pred = read_conll(args.pred_path)

    # output
    print('writing predictions to', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for i_sent in tqdm.tqdm(range(len(data))):
            f.write(data[i_sent]['name'] + '\n')
            for i_word, word in enumerate(data[i_sent]['words']):
                f.write('{:<30}{:<30}{:<30}\n'.format(word, data[i_sent]['labels'][i_word], pred[i_sent]['labels'][i_word]))
            f.write('\n\n')


if __name__ == '__main__':
    main()

