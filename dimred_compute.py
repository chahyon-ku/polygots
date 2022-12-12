import argparse
import json
import sklearn.manifold
import numpy as np
import os
import tqdm


def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--embed_dir', type=str, default='stsb/en-train-mulda-partial/')

    # dimred
    parser.add_argument('--dimred', type=str, default='pca', choices = ('pca', 'tsne'))

    # output
    parser.add_argument('--output_path', type=str, default='pca/en-train-mulda-partial.txt')
    args = parser.parse_args()

    samples_embeddings = []
    ids = []
    names_id = {}
    for dir_entry in tqdm.tqdm(os.scandir(args.embed_dir)):
        if dir_entry.name.endswith('.txt'):
            name = dir_entry.name.split('.')[0]
            id = len(names_id)
            names_id[name] = id
            embeddings = np.loadtxt(dir_entry.path)
            samples_embeddings.append(embeddings)
            ids.append(id * np.ones(len(embeddings)))
    
    samples_embeddings = np.concatenate(samples_embeddings, 0)
    ids = np.concatenate(ids, 0)
    
    if args.dimred == 'pca':
        dimreds = sklearn.decomposition.PCA(n_components=2).fit_transform(samples_embeddings)
    elif args.dimred == 'tsne':
        dimreds = sklearn.manifold.TSNE(n_components=2, n_jobs=-1).fit_transform(samples_embeddings)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump({'dimreds': dimreds.tolist(), 'ids': ids.tolist(), 'names': list(names_id.keys())}, f)


if __name__ == '__main__':
    main()
    