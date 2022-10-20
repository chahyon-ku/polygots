# polygots

1. Install AWS CLI (https://aws.amazon.com/cli/)
2. Download data (aws s3 cp --no-sign-request --recursive s3://multiconer/multiconer2022 ./data/)
3. Install dependencies (python=3.9, torch, transformers, pytorch_lightning, allennlp)
4. Integration of 2 baseline methods on the way (KB-NER-main is DAMO-NLP's, and multiconer-baseline is the xlm-roberta baseline)