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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_data', type=str, default='data/fr-en-helsinki.conll')
    # parser.add_argument('--train_data', type=str, default='data/MultiCoNER2/es-train.conll')
    parser.add_argument('--valid_data', type=str, default='data/MultiCoNER2/en-dev.conll')
    # parser.add_argument('--train_data', type=str, default='data/MultiCoNER/EN-English/en_train.conll')
    # parser.add_argument('--valid_data', type=str, default='data/MultiCoNER/EN-English/en_dev.conll')
    parser.add_argument('--data_version', type=str, default=2, choices=(1, 2))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # model
    parser.add_argument('--encoder_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--cache_dir', type=str, default='cache')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # train
    parser.add_argument('--log_dir', type=str, default='logs/'+datetime.datetime.now().strftime('%m-%d %H-%M'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--f_valid', type=int, default=1)
    parser.add_argument('--f_save', type=int, default=99999)
    args = parser.parse_args()

    # data
    target_vocab = lib.dataset.tags_v1 if args.data_version == 1 else lib.dataset.tags_v2
    train_data = lib.dataset.CoNLLDataset(args.train_data, target_vocab, args.encoder_model, args.cache_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=train_data.collate_batch,
                                               pin_memory=True)
    valid_data = lib.dataset.CoNLLDataset(args.valid_data, target_vocab, args.encoder_model, args.cache_dir)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=valid_data.collate_batch,
                                               pin_memory=True)

    # model
    os.makedirs(args.cache_dir, exist_ok=True)
    model = lib.model.NERModel(args.encoder_model, args.cache_dir, target_vocab, dropout_rate=0.1, eos_token_id=0, pad_token_id=1)
    model = model.to(args.device)

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)
    if args.resume is not None:
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts['model'])
        optim.load_state_dict(state_dicts['optim'])

    global_step = 0
    epoch_postfix = collections.OrderedDict({})
    epoch_tqdm = tqdm.tqdm(range(args.n_epochs))
    best_valid_loss = 999999
    for epoch in epoch_tqdm:
        model.reset_metrics()
        train_tqdm = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i_train, batch in train_tqdm:
            epoch_tqdm.set_postfix(epoch_postfix)
            tokens, tags, mask, token_mask, eos_masks, metadata = [e.to(args.device) if i_e < 5 else e for i_e, e in enumerate(batch)]

            optim.zero_grad()
            token_scores = model(tokens, mask)
            output = model.compute_results(token_scores, eos_masks, tags, metadata)
            output['loss'].backward()
            optim.step()

            with torch.no_grad():
                epoch_postfix['train_loss'] = output['loss'].item()
                epoch_postfix['train_f1'] = output['results']['MD-F1']
        summary_writer.add_scalar('train_loss', epoch_postfix['train_loss'], global_step)
        # summary_writer.add_scalar('train_f1', epoch_postfix['train_f1'], global_step)
        for metric_name, metric_value in output['results'].items():
            summary_writer.add_scalar(f'train_{metric_name}', metric_value, global_step)
        summary_writer.flush()
        global_step += 1

        with torch.no_grad():
            if (epoch + 1) % args.f_valid == 0:
                model.reset_metrics()
                valid_losses = []
                for i_train, batch in enumerate(valid_loader):
                    tokens, tags, mask, token_mask, eos_masks, metadata = [e.to(args.device) if i_e < 5 else e for i_e, e in enumerate(batch)]

                    token_scores = model(tokens, mask)
                    output = model.compute_results(token_scores, eos_masks, tags, metadata)

                    valid_losses.append(output['loss'].item())
                epoch_postfix['valid_loss'] = numpy.mean(valid_losses)
                epoch_postfix['valid_f1'] = output['results']['MD-F1']
                summary_writer.add_scalar('valid_loss', epoch_postfix['valid_loss'], global_step)
                # summary_writer.add_scalar('valid_f1', epoch_postfix['valid_f1'], global_step)
                for metric_name, metric_value in output['results'].items():
                    summary_writer.add_scalar(f'valid_{metric_name}', metric_value, global_step)

                summary_writer.flush()

            
            os.makedirs(args.log_dir, exist_ok=True)
            torch.save(model.state_dict(),
                        os.path.join(args.log_dir, f'last.pt'))

            if epoch_postfix['valid_loss'] < best_valid_loss:
                torch.save(model.state_dict(),
                           os.path.join(args.log_dir, f'best.pt'))

            if (epoch + 1) % args.f_save == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.log_dir, f'{epoch}.pt'))
            train_tqdm.reset()
