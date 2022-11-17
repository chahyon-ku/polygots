import argparse
import collections
import json
import os

import numpy as np
import tensorboardX
import torch
import transformers

import lib
import tqdm


def post_process(s, tag_list):
    for tag in tag_list:
        if tag != 'O':
            s = s.replace(tag, f' {tag} ')
    s = s.replace('<s>', '')
    s = s.replace('</s>', '')
    s = s.replace('<pad>', '')
    s = ' '.join(s.split())
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='xlm-clm-enfr-1024', choices=('xlm-clm-enfr-1024'))

    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--device', type=str, default='cuda', choices=('cpu', 'cuda'))

    # generate
    parser.add_argument('--model_path', type=str, default='logs/clm/clm_en_40/39.pt')
    parser.add_argument('--output_dir', type=str, default='output/clm/en')
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    additional_special_tokens = list(lib.dataset.tags_v2.keys())[1:]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
                                                           additional_special_tokens=additional_special_tokens)

    model = lib.model.get_clm(args.model_name, args.cache_dir, args.device)
    model.load_state_dict(torch.load(args.model_path))

    with torch.no_grad():
        generated = []
        input_ids = tokenizer.encode('Hello, my dog is cute and ', return_tensors='pt').to(args.device)
        print(input_ids)
        beam_outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=4,
            num_return_sequences=5,
            early_stopping=True
        )
        for beam_output in beam_outputs:
            print(tokenizer.decode(beam_output, skip_special_tokens=True))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'samples.json'), 'w', encoding='utf-8') as f:
        json.dump(generated, f, indent=1)


if __name__ == '__main__':
    main()
