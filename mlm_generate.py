import argparse
import collections
import json
import os

import numpy as np
import tensorboardX
import torch
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
    parser.add_argument('--data_path', type=str, default='data/MultiCoNER2/en-train.conll')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mlm_prob', type=float,  default=0.15)

    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', choices=('xlm-roberta-base',
                                                                                       'xlm-roberta-large'))
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--device', type=str, default='cuda', choices=('cpu', 'cuda'))

    # generate
    parser.add_argument('--model_path', type=str, default='logs/melm/melm_en_40/39.pt')
    parser.add_argument('--output_dir', type=str, default='output/melm/en_incre_40')
    args = parser.parse_args()

    # data
    # data = lib.dataset.MLMDatasetv2(args.data_path, lib.dataset.tags_v2, args.model_name, args.cache_dir, args.mlm_prob,
    #                                 True)
    # loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
    #                                      num_workers=args.num_workers, collate_fn=data.collate_batch,
    #                                      pin_memory=True)
    #
    # model = lib.model.get_mlm(args.model_name, args.cache_dir, args.device)
    # model.load_state_dict(torch.load(args.model_path))
    #
    # tag_list = list(lib.dataset.tags_v2.keys())
    #
    # with torch.no_grad():
    #     generated = []
    #
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     data_tqdm = tqdm.tqdm(enumerate(loader), total=len(loader), leave=False)
    #     for i_batch, batch in data_tqdm:
    #         batch = [e.to(args.device) for e in batch]
    #         input_ids, attention_mask, labels = batch
    #
    #         initial_input_ids = input_ids.clone()
    #         for i in range(50):
    #             masked_indices = torch.where(input_ids == data.tokenizer.mask_token_id)
    #             if len(masked_indices[0]) == 0:
    #                 break
    #
    #             output = model(input_ids, attention_mask)
    #
    #             masked_indices_1 = torch.unique(masked_indices[0]).long()
    #             masked_indices_2 = torch.zeros(len(masked_indices_1)).long()
    #             for i, e in enumerate(masked_indices_1):
    #                 tmp = masked_indices[1][masked_indices[0] == e]
    #                 masked_indices_2[i] = tmp[np.random.randint(0, len(tmp))].item()
    #             masked_indices = (masked_indices_1, masked_indices_2)
    #
    #             k = 2
    #             masked_logits = output.logits[masked_indices]
    #             _, masked_topk_ids = torch.topk(masked_logits, k)
    #             input_ids[masked_indices] = masked_topk_ids[:, 0]
    #
    #         for i_sample, sample_ids in enumerate(input_ids):
    #             source_str = data.tokenizer.decode(labels[i_sample])
    #             source_str = post_process(source_str, tag_list)
    #             masked_str = data.tokenizer.decode(initial_input_ids[i_sample])
    #             masked_str = post_process(masked_str, tag_list)
    #             sample_str = data.tokenizer.decode(sample_ids)
    #             sample_str = post_process(sample_str, tag_list)
    #             generated.append({'source': source_str, 'masked': masked_str, 'sample': sample_str})
    #
    # os.makedirs(args.output_dir, exist_ok=True)
    # with open(os.path.join(args.output_dir, 'samples.json'), 'w', encoding='utf-8') as f:
    #     json.dump(generated, f, indent=1)

    with open(os.path.join(args.output_dir, 'samples.json'), 'r', encoding='utf-8') as f:
        generated = json.load(f)

    with open(os.path.join(args.output_dir, 'melm.conll'), 'w', encoding='utf-8') as f:
        for i_sample, sample in tqdm.tqdm(enumerate(generated)):
            f.write('# id \n')
            f.write(unlinearize(sample['sample']) + '\n')


if __name__ == '__main__':
    main()
