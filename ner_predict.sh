# for s in base/en midterm/en-partial midterm/en-full midterm/en-stable-full midterm/en-stable-partial final/en-helsinki final/en-mulda-partial final/en-mulda-full
# do
#     python ner_predict.py --num_workers 8 --test_data data/MultiCoNER2/en-dev.conll --resume ./logs/$s/last.pt --device cuda:0 --output_path preds/last/$s.conll
# done


# for s in base/fr midterm/fr-partial midterm/fr-full midterm/fr-stable-full midterm/fr-stable-partial final/fr-t5 final/fr-mulda-partial final/fr-mulda-full
# do 
#     python ner_predict.py --num_workers 8 --test_data data/MultiCoNER2/fr-dev.conll --resume ./logs/$s/last.pt --device cuda:0 --output_path preds/last/$s.conll
# done


# for s in kb/en-train-kb kb/en-mulda-partial-kb kb/en-mulda-full-kb
# do
#     python ner_predict.py --num_workers 8 --test_data data/MultiCoNER2/en-dev.conll --resume ./logs/$s/last.pt --device cuda:1 --output_path preds/last/orig/$s.conll
# done


for s in kb/en-train-kb kb/en-mulda-partial-kb kb/en-mulda-full-kb
do
    python ner_predict.py --num_workers 8 --test_data data/kb/en-pred-kb.conll --resume ./logs/$s/last.pt --device cuda:1 --output_path preds/last/$s.conll
done


# for s in kb/fr-train-kb kb/fr-mulda-partial-kb kb/fr-mulda-full-kb
# do 
#     python ner_predict.py --num_workers 8 --test_data data/MultiCoNER2/fr-dev.conll --resume ./logs/$s/last.pt --device cuda:1 --output_path preds/last/orig/$s.conll
# done


for s in kb/fr-train-kb kb/fr-mulda-partial-kb kb/fr-mulda-full-kb
do 
    python ner_predict.py --num_workers 8 --test_data data/kb/fr-pred-kb.conll --resume ./logs/$s/last.pt --device cuda:1 --output_path preds/last/$s.conll
done