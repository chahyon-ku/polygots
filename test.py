import argparse
import collections
import datetime
import os
import numpy
import tensorboardX
import torch
from torch.utils.data import DataLoader
import lib
import tqdm


wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_data', type=str, default='data/MultiCoNER2/en-dev.conll')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # model
    parser.add_argument('--encoder_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--cache_dir', type=str, default='cache')

    # train
    parser.add_argument('--resume', type=str, default='logs/10-25 12-10/10.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # data
    test_data = lib.dataset.CoNLLDataset(args.test_data, wnut_iob, args.encoder_model, args.cache_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=test_data.collate_batch,
                                               pin_memory=True)

    # model
    os.makedirs(args.cache_dir, exist_ok=True)
    model = lib.model.NERModel(args.encoder_model, args.cache_dir, wnut_iob)
    model = model.cuda()

    # train
    if args.resume is not None:
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts['model'])
        optim.load_state_dict(state_dicts['optim'])

    test_tqdm = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    with torch.no_grad():
        losses = []
        results = []
        for i_batch, batch in test_tqdm:
            tokens, tags, mask, token_mask, metadata = [e.to(args.device) if i_e < 4 else e for i_e, e in enumerate(batch)]

            token_scores = model(tokens, mask)
            output = model.compute_results(token_scores, mask, tags, metadata)

            losses.append()
            epoch_postfix['train_loss'] = output['loss'].item()
            epoch_postfix['train_f1'] = output['result']['MD@F1'].item()
            summary_writer.add_scalar('train_loss', epoch_postfix['train_loss'], global_step)
            summary_writer.add_scalar('train_f1', epoch_postfix['train_f1'], global_step)
            for metric_name, metric_value in output['results'].items():
                summary_writer.add_scalar(metric_name, metric_value, global_step)
            global_step += 1
