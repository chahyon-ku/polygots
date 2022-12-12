# python ner_train.py --num_workers 8 --train_data data/MultiCoNER2/en-train.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/base/en --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/MultiCoNER2/fr-train.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/base/fr --n_epochs 20 --device cuda:0

# python ner_train.py --num_workers 8 --train_data data/midterm/en-partial.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/midterm/en-partial --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/en-full.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/midterm/en-full --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/en-stable-full.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/midterm/en-stable-full --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/en-stable-partial.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/midterm/en-stable-partial --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/fr-partial.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/midterm/fr-partial --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/fr-full.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/midterm/fr-full --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/fr-stable-full.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/midterm/fr-stable-full --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/midterm/fr-stable-partial.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/midterm/fr-stable-partial --n_epochs 20 --device cuda:0

# python ner_train.py --num_workers 8 --train_data data/final/en-helsinki.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/en-helsinki --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/final/en-mulda-partial.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/en-mulda-partial.conll --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/final/en-mulda-full.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/en-mulda-full --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/final/fr-t5.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/fr-t5 --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/final/fr-mulda-partial.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/fr-mulda-partial --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/final/fr-mulda-full.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/fr-mulda-full --n_epochs 20 --device cuda:1

# python ner_train.py --num_workers 8 --train_data data/kb/fr-train-kb.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/kb/fr-train-kb --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/kb/fr-mulda-partial-kb.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/kb/fr-mulda-partial-kb --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/kb/fr-mulda-full-kb.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/kb/fr-mulda-full-kb --n_epochs 20 --device cuda:1
# python ner_train.py --num_workers 8 --train_data data/kb/en-train-kb.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/kb/en-train-kb --n_epochs 20 --device cuda:0
# python ner_train.py --num_workers 8 --train_data data/kb/en-mulda-partial-kb.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/kb/en-mulda-partial-kb --n_epochs 20 --device cuda:0
python ner_train.py --num_workers 8 --train_data data/kb/en-mulda-full-kb.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/kb/en-mulda-full-kb --n_epochs 20 --device cuda:0