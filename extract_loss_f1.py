import argparse
import json
import os
import time

import tensorboardX
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='logs/transv2_loss_f1_p_r')
    parser.add_argument('--output_dir', type=str, default='logs/transv2_loss_f1_p_r_maxes')
    args = parser.parse_args()

    for dir_entry in os.scandir(args.input_dir):
        for event_entry in os.scandir(dir_entry.path):
            if event_entry.name.startswith('events'):
                print(event_entry.path)
                start = time.time()
                ea = event_accumulator.EventAccumulator(event_entry.path, size_guidance={event_accumulator.SCALARS: 0,})
                ea.Reload()
                print('Reload took', time.time() - start)

                start = time.time()
                os.makedirs(os.path.join(args.output_dir, dir_entry.name), exist_ok=True)
                sw = tensorboardX.SummaryWriter(os.path.join(args.output_dir, dir_entry.name))
                maxes = {'valid_MD-F1': 0, 'valid_MD-P': 0, 'valid_MD-R': 0}
                max_step = -1
                for tag in ['train_loss', 'train_MD-F1', 'valid_loss', 'valid_MD-F1', 'valid_MD-P', 'valid_MD-R']:
                    for scalar_event in ea.Scalars(tag):
                        # [wall_time, step, value]
                        if tag == 'valid_MD-F1':
                            if maxes[tag] < scalar_event[2]:
                                max_step = scalar_event[1]
                                maxes[tag] = scalar_event[2]
                        elif scalar_event[1] == max_step:
                            maxes[tag] = scalar_event[2]
                        sw.add_scalar(tag, scalar_event[2], scalar_event[1], scalar_event[0])
                sw.flush()
                print('Write took', time.time() - start)
                with open(os.path.join(args.output_dir, dir_entry.name, 'maxes.txt'), 'w') as f:
                    json.dump(maxes, f)


if __name__ == '__main__':
    main()
