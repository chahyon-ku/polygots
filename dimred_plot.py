import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--input_path', type=str, default='pca/fr-train-stable-partial.txt')

    # sample
    parser.add_argument('--n_samples', type=int, default=100)

    # output
    parser.add_argument('--output_path', type=str, default='pca/fr-train-stable-partial.png')
    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        dimred_id_name = json.load(f)

    dimreds = np.array(dimred_id_name['dimreds'])
    ids = np.array(dimred_id_name['ids'])
    names = dimred_id_name['names']
    markers = ['o', 'v', '^', '<', '>', 's', 'p']

    s=100
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    for id in np.unique(ids):
        indices = np.where(ids == id)[0]
        print(indices)
        indices = np.random.choice(indices, args.n_samples, replace=False)
        plt.scatter(dimreds[indices, 0], dimreds[indices, 1], s=s, marker=markers[int(id)], label=ids[indices])
    plt.legend(names)
    plt.title('all')
    plt.savefig(args.output_path)
    plt.close()

    # plt.show()


if __name__ == '__main__':
    main()