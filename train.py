import argparse
import lib
import transformers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    lib.ner_model.NERBaseAnnotator()