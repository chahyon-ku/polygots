import argparse
import json
import tqdm
import lib.dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_path', type=str, default='output/translation/en-fr/results.txt')
    args = parser.parse_args()

    with open(args.trans_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    n_errors = 0
    tag_list = list(lib.dataset.tags_v2.keys())
    for result in tqdm.tqdm(results):
        input = result['input']
        trans = result['translatedText']
        for tag in tag_list:
            if tag != 'O':
                input_count = input.count(tag)
                input_span_count = input.count(f'notranslate">{tag}</span>')
                trans_count = trans.count(tag)
                trans_span_count = trans.count(f'notranslate">{tag}</span>')
                if trans_count != trans_span_count:
                    print('span disappeared from', input)
                    n_errors += 1
                if input_count != trans_count:
                    print(f'{tag} disappeared from', input)
                    n_errors += 1
    print(n_errors)


if __name__ == '__main__':
    main()