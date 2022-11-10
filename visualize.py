import argparse
import os
import tqdm


def read_conll(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            elif line[0] == '#':
                data.append({'name': line, 'words': [], 'labels': []})
            else:
                words = line.split(' _')[0].strip()
                labels = line.split(' _')[2].strip()
                data[-1]['words'].append(words)
                data[-1]['labels'].append(labels)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/MultiCoNER2/en-dev.conll')
    parser.add_argument('--pred_path', type=str, default='output/base_coner2_en_40/en-dev-pred.conll')
    parser.add_argument('--output_path', type=str, default='output/base_coner2_en_40/en-dev-diff.conll')
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

