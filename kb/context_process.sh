# python context_process.py --retrieval_file v1/fr-train.txt --conll_file datasets/fr/fr-train.conll --lang fr
# python context_process.py --retrieval_file v1/fr-dev.txt --conll_file datasets/fr/fr-dev.conll --lang fr
# python context_process.py --retrieval_file v1/fr-orig-mulda.txt --conll_file datasets/fr/fr-orig-mulda.conll --lang fr
# python context_process.py --retrieval_file v1/fr-trans-mulda.txt --conll_file datasets/fr/fr-trans-mulda.conll --lang fr

# python context_process.py --retrieval_file v1/en-train.txt --conll_file datasets/en/en-train.conll --lang en
# python context_process.py --retrieval_file v1/en-dev.txt --conll_file datasets/en/en-dev.conll --lang fr
# python context_process.py --retrieval_file v1/en-orig-mulda.txt --conll_file datasets/en/en-orig-mulda.conll --lang en
# python context_process.py --retrieval_file v1/en-trans-mulda.txt --conll_file datasets/en/en-trans-mulda.conll --lang en

python context_process.py --retrieval_file v1/en.txt --conll_file datasets/en/en.conll --lang en
python context_process.py --retrieval_file v1/fr.txt --conll_file datasets/fr/fr.conll --lang fr