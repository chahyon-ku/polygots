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
    parser.add_argument('--train_data', type=str, default='data/MultiCoNER2/en-train.conll')
    parser.add_argument('--valid_data', type=str, default='data/MultiCoNER2/en-dev.conll')
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
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--f_valid', type=int, default=1)
    parser.add_argument('--f_save', type=int, default=1)
    args = parser.parse_args()

    # data
    train_data = lib.dataset.CoNLLDataset(args.train_data, wnut_iob, args.encoder_model, args.cache_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=train_data.collate_batch,
                                               pin_memory=True)
    valid_data = lib.dataset.CoNLLDataset(args.valid_data, wnut_iob, args.encoder_model, args.cache_dir)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=valid_data.collate_batch,
                                               pin_memory=True)

    # model
    os.makedirs(args.cache_dir, exist_ok=True)
    model = lib.model.NERModel(args.encoder_model, args.cache_dir, wnut_iob)
    model = model.cuda()

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # warmup_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1,
    #                                               total_iters=warmup_steps)
    # train_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-6,
    #                                              total_iters=total_steps - warmup_steps)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_lr, train_lr], [warmup_steps])

    # train
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)
    if args.resume is not None:
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts['model'])
        optim.load_state_dict(state_dicts['optim'])

    global_step = 0
    epoch_postfix = collections.OrderedDict({})
    epoch_tqdm = tqdm.tqdm(range(args.n_epochs))
    for epoch in epoch_tqdm:
        model.reset_metrics()
        train_tqdm = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i_train, batch in train_tqdm:
            epoch_tqdm.set_postfix(epoch_postfix)
            tokens, tags, mask, token_mask, metadata = [e.to(args.device) if i_e < 4 else e for i_e, e in enumerate(batch)]

            optim.zero_grad()
            token_scores = model(tokens, mask)
            output = model.compute_results(token_scores, mask, tags, metadata)
            output['loss'].backward()
            optim.step()

            with torch.no_grad():
                epoch_postfix['train_loss'] = output['loss'].item()
                epoch_postfix['train_f1'] = output['results']['MD-F1']
                summary_writer.add_scalar('train_loss', epoch_postfix['train_loss'], global_step)
                summary_writer.add_scalar('train_f1', epoch_postfix['train_f1'], global_step)
                for metric_name, metric_value in output['results'].items():
                    summary_writer.add_scalar(metric_name, metric_value, global_step)
                global_step += 1

        with torch.no_grad():
            if (epoch + 1) % args.f_valid == 0:
                model.reset_metrics()
                valid_losses = []
                for i_train, batch in enumerate(valid_loader):
                    tokens, tags, mask, token_mask, metadata = [e.to(args.device) if i_e < 4 else e for i_e, e in enumerate(batch)]

                    token_scores = model(tokens, mask)
                    output = model.compute_results(token_scores, mask, tags, metadata)

                    valid_losses.append(output['loss'].item())
                epoch_postfix['valid_loss'] = numpy.mean(valid_losses)
                epoch_postfix['valid_f1'] = output['results']['MD-F1'].item()
                summary_writer.add_scalar('valid_loss', epoch_postfix['valid_loss'], global_step)
                for metric_name, metric_value in output['results'].items():
                    summary_writer.add_scalar(metric_name, metric_value, global_step)

            if (epoch + 1) % args.f_save == 0:
                os.makedirs(args.log_dir, exist_ok=True)
                torch.save({'model': model.state_dict(), 'optim': optim.state_dict()},
                           os.path.join(args.log_dir, f'{epoch}.pt'))
            train_tqdm.reset()
