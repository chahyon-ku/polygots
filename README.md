# Folder Structure

    .
    ├── data                    # Data CONLL files (No KB-NER data due to +100MB limit)
    │   ├── augment_only        # Augmented data only
    │   ├── final               # MulDA, T5, and Helsinki data
    │   └── midterm             # Our translation data
    ├── kb                      # Knowledge base augmentation files (adapted from KB-NER https://github.com/Alibaba-NLP/KB-NER)
    ├── lib                     # Class and function definitions (adapted from MultiCoNER I baseline https://github.com/amzn/multiconer-baseline)
    │   ├── dataset.py          # PyTorch dataset class
    │   ├── ...
    │   ├── metric.py           # Metric calculation
    │   ├── model.py            # PyTorch model class
    │   └── reader_utils.py     # CONLL reader
    ├── olds                    # Contains dropped files including clm, mlm, and linearized translate
    ├── logs                    # Training logs
    ├── preds                   # Predictions and evaluation results
    ├── ner_train.sh            # All training scripts
    ├── ner_predict.sh          # All evaluation scripts
    ├── ner_results.py          # Organize results into ner_results.txt
    ├── ner_results.txt         # Generated results
    ├── stsb_predict.py         # Save Sentence-BERT embeddings
    ├── dimred_compute.sh       # Compute dimensionality reduction of Sentence-BERT embeddings (pca, tsne)
    ├── dimred_plot.sh          # Plot dimensionality reduction of Sentence-BERT embeddings (pca, tsne)
    ├── Error analysis.ipynb    # Error analysis
    ├── README.md               # This file
    └── ...