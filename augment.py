import argparse

import lib.reader_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='data/MultiCoNER2/en-train.conll')
    parser.add_argument('--dst_path', type=str, default='data/en-aug.conll')
    args = parser.parse_args()

    print(lib.reader_utils.get_ner_reader(args.src_path))


if __name__ == '__main__':
    main()