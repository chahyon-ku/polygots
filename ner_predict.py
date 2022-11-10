import argparse
import collections
import os
import torch
from torch.utils.data import DataLoader
import lib
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_data', type=str, default='data/MultiCoNER2/en-dev.conll')
    parser.add_argument('--data_version', type=str, default=2, choices=(1, 2))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    # model
    parser.add_argument('--encoder_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--cache_dir', type=str, default='cache')

    # predict
    parser.add_argument('--resume', type=str, default='logs/base_coner2_en_40/39.pt')
    parser.add_argument('--device', type=str, default='cuda')

    # output
    parser.add_argument('--output_path', type=str, default='output/base_coner2_en_40/en-dev-pred.conll')
    args = parser.parse_args()

    # data
    conll = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            elif line[0] == '#':
                conll.append({'name': line, 'words': [], 'labels': []})
            else:
                words = line.split(' _')[0].strip()
                labels = line.split(' _')[2].strip()
                conll[-1]['words'].append(words)
                conll[-1]['labels'].append(labels)
    print('conll built')

    tags_to_id = lib.dataset.tags_v1 if args.data_version == 1 else lib.dataset.tags_v2
    id_to_tags = {v: k for k, v in tags_to_id.items()}
    test_data = lib.dataset.CoNLLDataset(args.test_data, tags_to_id, args.encoder_model, args.cache_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, collate_fn=test_data.collate_batch,
                                              pin_memory=True)

    # model
    os.makedirs(args.cache_dir, exist_ok=True)
    model = lib.model.NERModel(args.encoder_model, args.cache_dir, tags_to_id)
    model = model.cuda()

    # test
    if args.resume is not None:
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts['model'])

    with torch.no_grad():
        losses = []
        f1s = []
        i_sample = 0
        test_postfix = collections.OrderedDict({})
        test_tqdm = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        for i_batch, batch in test_tqdm:
            test_tqdm.set_postfix(test_postfix)
            tokens, tags, mask, token_mask, metadata = [e.to(args.device) if i_e < 4 else e for i_e, e in enumerate(batch)]

            token_scores = model(tokens, mask)
            output = model.compute_results(token_scores, mask, tags, metadata, 'predict')
            for i_sent, token_tags in enumerate(output['token_tags']):
                s_tokens = test_data.tokenizer.convert_ids_to_tokens(tokens[i_sent])
                conll[i_sample]['preds'] = [tag for i_token, tag in enumerate(token_tags) if s_tokens[i_token][0] == '▁']
                assert(len(conll[i_sample]['words']) == len(conll[i_sample]['labels']) == len(conll[i_sample]['preds']))
                i_sample += 1

            test_postfix['test_loss'] = output['loss'].item()
            test_postfix['test_f1'] = output['results']['MD-F1']
            losses.append(test_postfix['test_loss'])
            f1s.append(test_postfix['test_f1'])
        print(f'mean test loss: {sum(losses) / len(f1s):.6f}, mean test f1: {sum(f1s) / len(f1s):.6f}')

    # output
    print('writing predictions to', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for conll_one in tqdm.tqdm(conll):
            f.write(conll_one['name'] + '\n')
            for i_word, word in enumerate(conll_one['words']):
                f.write('{} _ _ {}\n'.format(word, conll_one['preds'][i_word]))
            f.write('\n\n')
