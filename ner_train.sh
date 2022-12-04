#python ner_train.py --num_workers 8 --train_data data/en-asal.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/base_en-asal_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/en-ku.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/base_en-ku_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/en-london.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/base_en-london_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/fr-asal.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/base_fr-asal_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/fr-ku.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/base_fr-ku_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/fr-london.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/base_fr-london_20 --n_epochs 20


#python ner_train.py --num_workers 8 --train_data data/en-orig-london.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/transv2/base_en-orig-london_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/fr-orig-london.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/transv2/base_fr-orig-london_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/MultiCoNER2/en-train.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/transv2/base_en_20 --n_epochs 20
#python ner_train.py --num_workers 8 --train_data data/MultiCoNER2/fr-train.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/transv2/base_fr_20 --n_epochs 20

# python ner_train.py --num_workers 8 --train_data data/en-melm.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/transv2/base_en-melm_20 --n_epochs 20
# python ner_train.py --num_workers 8 --train_data data/fr-melm.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/transv2/base_fr-melm_20 --n_epochs 20

python ner_train.py --num_workers 8 --train_data data/fr-en-helsinki.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/fr-en-helsinki --n_epochs 20
python ner_train.py --num_workers 8 --train_data data/fr-en-orig-mulda.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/fr-en-orig-mulda --n_epochs 20
python ner_train.py --num_workers 8 --train_data data/fr-en-trans-mulda.conll --valid_data data/MultiCoNER2/en-dev.conll --log_dir ./logs/final/fr-en-trans-mulda --n_epochs 20
python ner_train.py --num_workers 8 --train_data data/en-fr-orig-mulda.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/en-fr-orig-mulda --n_epochs 20
python ner_train.py --num_workers 8 --train_data data/en-fr-t5.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/en-fr-t5 --n_epochs 20
python ner_train.py --num_workers 8 --train_data data/en-fr-trans-mulda.conll --valid_data data/MultiCoNER2/fr-dev.conll --log_dir ./logs/final/en-fr-trans-mulda --n_epochs 20