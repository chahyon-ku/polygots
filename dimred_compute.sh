python dimred_compute.py --embed_dir stsb/en-train-mulda-partial/ --dimred pca --output_path pca/en-train-mulda-partial.txt
python dimred_compute.py --embed_dir stsb/fr-train-stable-partial/ --dimred pca --output_path pca/fr-train-stable-partial.txt

python dimred_compute.py --embed_dir stsb/en-train-mulda-partial/ --dimred tsne --output_path tsne/en-train-mulda-partial.txt
python dimred_compute.py --embed_dir stsb/fr-train-stable-partial/ --dimred tsne --output_path tsne/fr-train-stable-partial.txts