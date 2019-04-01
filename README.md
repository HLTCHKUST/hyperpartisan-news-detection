# hyperpartisan-news-detection
This repository is for the paper in SemEval 2019 Task 4: Hyperpartisan News Detection (Hyperpartisan News Detection by de-noising weakly-labeled data)

## Fine-tune BERT Language Model
Process hyperpartisan news dataset first (for BERT language model training).

```
python process_data_for_bert_training.py
```

Use processed hyperpartisan news articles to train BERT language model. (run_lm_finetuning.py comes from https://github.com/huggingface/pytorch-pretrained-BERT)

```
python run_lm_finetuning.py --train_file=data_new/article_corpus.txt --output_dir=bert_model --bert_model=bert-base-uncased --do_train --on_memory
```

## Train BERT + Classifier for denoising
Use by-article data to train Classifier (BERT LM model is freezed) for denoising by-publisher data
```
python main --do_train --use_bert --batch_size=16
```

## Train BERT + LSTM + Classifier by denoised by-publisher data
```
python main.py --do_train --train_cleaner_dataset --hidden_dim=300 --hidden_dim_tit=100 --batch_size=16 --weight_decay=1e-6
```

## Test model on by-article data
```
python main.py --do_eval_bert_plus_lstm --train_cleaner_dataset --hidden_dim=300 --hidden_dim_tit=100 --batch_size=16 
```
