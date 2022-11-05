import collections

if __name__ == '__main__':
    tags_count = collections.defaultdict(int)
    with open('data/MultiCoNER2/de-train.conll', 'r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            line = line.strip()
            if len(line) and '_' in line:
                tag = line.split(' _')[2].strip()
                tags_count[tag] += 1
    pairs = [(k, v) for k, v in sorted(tags_count.items(), key=lambda item: item[1], reverse=True)]
    print(pairs)
    print({k: i_kv for i_kv, (k, v) in enumerate(pairs)})
    print(len(tags_count))
