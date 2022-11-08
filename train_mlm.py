import argparse
import collections
import os

import torch
import lib
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/MultiCoNER2/de-train.conll')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mlm_prob', type=float,  default=0.15)

    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', choices=('xlm-roberta-base',
                                                                                       'xlm-roberta-large'))
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--device', type=str, default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--mode', type=str, default='daga', choices=('daga', 'melm', 'word'))

    parser.add_argument('--lr', type=float, default=1e-6)

    parser.add_argument('--dst_path', type=str, default='data/de-aug.conll')
    args = parser.parse_args()
    train_data = lib.dataset.MLMDataset(args.train_path, lib.dataset.tags_v2, args.model_name, args.cache_dir, args.mlm_prob)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=train_data.collate_batch,
                                               pin_memory=True)

    model = lib.model.get_mlm(args.model_name, args.cache_dir, args.device)
    optim = torch.optim.AdamW(model.parameters(), args.lr)

    train_postfix = collections.OrderedDict()
    train_tqdm = tqdm.tqdm(train_loader)
    for batch in train_tqdm:
        batch = [e.to(args.device) for e in batch]
        input_ids, attention_mask, labels = batch
        # input_ids[input_ids > 250001] = 250001
        # labels[labels > 250001] = 250001

        optim.zero_grad()
        output = model(input_ids, attention_mask, labels=labels)
        output.loss.backward()
        optim.step()

        with torch.no_grad():
            train_postfix['train_loss'] = output.loss.item()
            train_tqdm.set_postfix(train_postfix)

    os.makedirs(args.log_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{args.log_dir}/model.pt')


if __name__ == '__main__':
    main()
