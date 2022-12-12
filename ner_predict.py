import argparse
import collections
import json
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
    parser.add_argument('--resume', type=str, default='logs/kb/fr-trans-mulda/best.pt')
    parser.add_argument('--device', type=str, default='cuda')

    # output
    parser.add_argument('--output_path', type=str, default='output/kb/fr-trans-mulda/en-dev-pred.conll')
    args = parser.parse_args()

    # data
    conll = []
    name = ''
    words = []
    labels = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
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

    tags_to_id = lib.dataset.tags_v1 if args.data_version == 1 else lib.dataset.tags_v2
    id_to_tags = {v: k for k, v in tags_to_id.items()}
    test_data = lib.dataset.CoNLLDataset(args.test_data, tags_to_id, args.encoder_model, args.cache_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, collate_fn=test_data.collate_batch,
                                              pin_memory=True)

    # model
    os.makedirs(args.cache_dir, exist_ok=True)
    model = lib.model.NERModel(args.encoder_model, args.cache_dir, tags_to_id, 0.1, 0, 1)
    model = model.to(args.device)

    # test
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    model.eval()
    with torch.no_grad():
        losses = []
        f1s = []
        i_sample = 0
        test_postfix = collections.OrderedDict({})
        test_tqdm = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        for i_batch, batch in test_tqdm:
            test_tqdm.set_postfix(test_postfix)
            tokens, tags, mask, token_mask, eos_masks, metadata = [e.to(args.device) if i_e < 5 else e for i_e, e in enumerate(batch)]

            token_scores = model(tokens, mask)
            # print(token_scores, metadata)
            output = model.compute_results(token_scores, eos_masks, tags, metadata, 'predict')
            for i_sent, token_tags in enumerate(output['token_tags']):
                s_tokens = test_data.tokenizer.convert_ids_to_tokens(tokens[i_sent])
                conll[i_sample]['preds'] = [tag for i_token, tag in enumerate(token_tags) if s_tokens[i_token][0] == '▁']
                if not len(conll[i_sample]['words']) == len(conll[i_sample]['labels']) >= len(conll[i_sample]['preds']):
                    print(conll[i_sample]['words'], conll[i_sample]['labels'], conll[i_sample]['preds'])
                assert(len(conll[i_sample]['words']) == len(conll[i_sample]['labels']) >= len(conll[i_sample]['preds']))
                i_sample += 1

            test_postfix['test_loss'] = output['loss'].item()
            test_postfix['test_f1'] = output['results']['MD-F1']
            losses.append(test_postfix['test_loss'])
            f1s.append(test_postfix['test_f1'])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # eval
    with open(args.output_path.replace('.conll', '.eval'), 'w', encoding='utf-8') as f:
        json.dump({'loss': sum(losses) / len(losses),
                   'f1': output['results']['MD-F1'],
                   'p': output['results']['MD-P'],
                   'r': output['results']['MD-R']}, f)

    # predict
    print('writing predictions to', args.output_path)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for conll_one in tqdm.tqdm(conll):
            f.write(conll_one['name'] + '\n')
            for i_pred, pred in enumerate(conll_one['preds']):
                f.write('{} _ _ {}\n'.format(conll_one['words'][i_pred], pred))
            f.write('\n\n')
