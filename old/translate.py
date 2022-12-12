import argparse
import html
import json
import os
from google.cloud import translate_v2 as translate
import tqdm
import lib.dataset


coner_tags_quotes = [f'\"{tag}' for tag in lib.dataset.tags_v2.keys()]
coner_tags_dash = [f'{tag}-' for tag in lib.dataset.tags_v2.keys()]


def preprocess(sentence):
    str = []
    for word, tag in sentence:
        if tag == "O":
            str.append(word)
        else:
            str.append(f'"<span class="notranslate">{tag}</span> {word}"')
    original_string = ' '.join(str)
    return original_string


def postprocess(sentence):
    t = sentence
    t = html.unescape(t)
    t = t.replace('<span class="notranslate">', '')
    t = t.replace('</span>', '')
    t = t.replace("\u201c", '"')
    t = t.replace("\u201d", '"')
    t = t.replace("\u201e", '"')
    t = t.replace("\u00bb", '"')
    t = t.replace("\u00ab", '"')
    t = t.replace(" DOCSTART ", "-DOCSTART-")
    for tag in lib.dataset.tags_v2.keys():
        t = t.replace(f'{tag}-', f'{tag} ')
    s = t.split()

    start = 0
    while start < len(s) - 1:
        if s[start] in coner_tags_quotes:
            n = 0
            end = start + 1
            for j in range(end, len(s)):
                if '\"' in s[j]:
                    end = j
                    break
            for j in range(end, start + 2, -1):
                s.insert(j, f'I-{s[start][3:]}')
                n += 1
            start = end + n
        else:
            start = start + 1

    s = ' '.join(s)
    s = s.replace("\"", "")
    s = ' '.join(s.split())

    return s


def unlinearize(sentence):
    unlin = ''
    i = 0
    s = sentence.split()
    while i < len(s):
        if s[i] in lib.dataset.tags_v2:
            if i < len(s) - 1:
                unlin += f'{s[i + 1]} _ _ {s[i]}\n'
            i += 2
        else:
            unlin += f'{s[i]} _ _ O\n'
            i += 1
    return unlin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='fr')
    parser.add_argument('--target', type=str, default='en')
    parser.add_argument('--source_dir', type=str, default='data/MultiCoNER2')
    parser.add_argument('--output_dir', type=str, default='output/translation2/')
    parser.add_argument('--account_json', type=str, default='google_api.json')
    args = parser.parse_args()

    translator = translate.Client.from_service_account_json(args.account_json)

    ids = []
    sentences = []
    with open(os.path.join(args.source_dir, f'{args.source}-train.conll'), 'r', encoding='utf-8') as inf:
        sentence = []
        for line in inf:
            line = line.strip()
            line = line.replace('_', '')
            if line != '':
                if line[0] == '#':
                    ids.append(line)
                else:
                    word_tag = line.split()
                    if len(word_tag) == 2:
                        sentence.append(word_tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
    # sentences = list(sentences[:10])
    sources = [preprocess(sentence) for sentence in sentences]

    results = []
    batch_size = 100
    for i in tqdm.tqdm(range(len(sources) // batch_size + 1)):
        start = batch_size * i
        end = min(batch_size * (i + 1), len(sources))
        results.extend(translator.translate(list(sources[start:end]), args.target, 'html', args.source))
    os.makedirs(os.path.join(args.output_dir, f'{args.source}-{args.target}'), exist_ok=True)
    with open(os.path.join(args.output_dir, f'{args.source}-{args.target}/results.txt'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(args.output_dir, f'{args.source}-{args.target}/results.txt'), 'r', encoding='utf-8') as f:
        results = json.load(f)

    trans = [(ids[i_result], postprocess(result['translatedText'])) for i_result, result in enumerate(results)]
    with open(os.path.join(args.output_dir, f'{args.source}-{args.target}/trans.txt'), 'w', encoding='utf-8') as f:
        json.dump(trans, f, indent=1)

    conll = [(id, unlinearize(trans)) for id, trans in trans]
    with open(os.path.join(args.output_dir, f'{args.source}-{args.target}/{args.source}-{args.target}.conll'), 'w', encoding='utf-8') as f:
        for id, unlin in conll:
            f.write(id + '\n')
            f.write(unlin + '\n')


if __name__ == '__main__':
    main()
