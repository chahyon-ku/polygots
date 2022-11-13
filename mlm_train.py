import argparse
import collections
import os

import numpy as np
import tensorboardX
import torch
import lib
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/MultiCoNER2/en-train.conll')
    parser.add_argument('--valid_path', type=str, default='data/MultiCoNER2/en-dev.conll')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mlm_prob', type=float,  default=0.15)

    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', choices=('xlm-roberta-base',
                                                                                       'xlm-roberta-large'))
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--device', type=str, default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--mode', type=str, default='daga', choices=('daga', 'melm', 'word'))

    parser.add_argument('--lr', type=float, default=1e-6)

    # train
    parser.add_argument('--log_dir', type=str, default='logs/mlm_en_20')
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--f_valid', type=int, default=1)
    parser.add_argument('--f_save', type=int, default=1)
    parser.add_argument('--dst_path', type=str, default='data/de-aug.conll')
    args = parser.parse_args()

    # data
    train_data = lib.dataset.MLMDataset(args.train_path, lib.dataset.tags_v2, args.model_name, args.cache_dir, args.mlm_prob,
                                        True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=train_data.collate_batch,
                                               pin_memory=True)
    valid_data = lib.dataset.MLMDataset(args.valid_path, lib.dataset.tags_v2, args.model_name, args.cache_dir, args.mlm_prob,
                                        True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=valid_data.collate_batch,
                                               pin_memory=True)

    model = lib.model.get_mlm(args.model_name, args.cache_dir, args.device)
    optim = torch.optim.AdamW(model.parameters(), args.lr)

    # train
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)

    global_step = 0
    epoch_postfix = collections.OrderedDict({})
    epoch_tqdm = tqdm.tqdm(range(args.n_epochs))
    for epoch in epoch_tqdm:
        train_tqdm = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i_train, batch in train_tqdm:
            batch = [e.to(args.device) for e in batch]
            input_ids, attention_mask, labels = batch

            optim.zero_grad()
            output = model(input_ids, attention_mask, labels=labels)
            output.loss.backward()
            optim.step()

            with torch.no_grad():
                epoch_postfix['train_loss'] = output.loss.item()
                epoch_tqdm.set_postfix(epoch_postfix)
                summary_writer.add_scalar('train_loss', epoch_postfix['train_loss'], global_step)
                summary_writer.flush()
                global_step += 1

        with torch.no_grad():
            if (epoch + 1) % args.f_valid == 0:
                valid_losses = []
                for i_train, batch in enumerate(valid_loader):
                    batch = [e.to(args.device) for e in batch]
                    input_ids, attention_mask, labels = batch
                    output = model(input_ids, attention_mask, labels=labels)

                    valid_losses.append(output.loss.item())
                    epoch_postfix['valid_loss'] = output.loss.item()
                    epoch_tqdm.set_postfix(epoch_postfix)
                epoch_postfix['valid_loss'] = np.mean(valid_losses)
                summary_writer.add_scalar('valid_loss', epoch_postfix['valid_loss'], global_step)
                summary_writer.flush()

            if (epoch + 1) % args.f_save == 0:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch}.pt'))
            train_tqdm.reset()


if __name__ == '__main__':
    main()
