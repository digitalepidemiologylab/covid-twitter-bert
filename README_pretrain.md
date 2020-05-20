# Pretraining

Our pretraining code starts from an existing pretrained model (such as BERT-Large) and several steps of unsupervised pretraining on data from a target domain (in our case Twitter data). This code can in principle be used for any domain specific pretraining.

## Prepare data
Data preparation is in two steps:

### Clean/preprocess data

The first step includes asciifying emojis, cleaning of usernames/URLs, etc. and can be run with the following script
```bash
cd preprocess
python prepare_pretrain_data.py \
  --input_data <path to folder containing .txt files> \
  --run_prefix <custom run prefix> \
  --model_class bert_large_uncased_wwm \
  --asciify_emojis \
  --username_filler twitteruser \
  --url_filler twitterurl \
  --replace_multiple_usernames \
  --replace_multiple_urls \
  --remove_unicode_symbols
```
This results in preprocessed data stored in `data/pretrain/<run_name>/preprocessed/`.

### Generate TFrecord files to be used for pretraining
```bash
cd preprocess
python create_pretrain_data.py \
  --run_name <generated run_name from before>
  --max_seq_length 96 \
  --dupe_factor 10 \
  --masked_lm_prob 0.15 \
  --short_seq_prob 0.1 \
  --model_class bert_large_uncased_wwm \
  --max_num_cpus 10
```
This process is potentially quite memory intensive and can take a long time, so choose max_num_cpus wisely ;). This results in preprocessed data stored in `data/pretrain/<run_name>/tfrecords/`.

You can then sync the data with your bucket:
```
cd data
gsutil -m rsync -u -r -x ".*.*/.*.txt$|.*<run name>/train/.*.tfrecords$" pretrain/ gs://<bucket name>/covid-bert/pretrain/pretrain_data/
```

## Run pretraining
Before you pretrain the models make sure to untar and copy the pretrained BERT-large model under `gs://cloud-tpu-checkpoints/bert/keras_bert/wwm_uncased_L-24_H-1024_A-16.tar.gz` to `gs://<bucket_name>/pretrained_models/bert/keras_bert/wwm_uncased_L-24_H-1024_A-16/`.

After the model and TFrecord files are present on the bucket, the following pretrain script can be run on a Google cloud VM with access to a TPU & bucket (same zone).
```bash
PRETRAIN_DATA=                                 # Run name of pretrain data
RUN_PREFIX=                                    # Custom run prefix (optional)
BUCKET_NAME=                                   # Bucket name (without gs:// prefix)              
TPU_IP=                                        # TPU IP
MODEL_CLASS=bert_large_uncased_wwm
TRAIN_BATCH_SIZE=1024
EVAL_BATCH_SIZE=1024

python run_pretrain.py \
  --run_prefix $RUN_PREFIX \
  --bucket_name $BUCKET_NAME \
  --tpu_ip $TPU_IP \
  --pretrain_data $PRETRAIN_DATA \
  --model_class $MODEL_CLASS \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --num_epochs 1 \
  --max_seq_length 96 \
  --learning_rate 2e-5 \
  --end_lr 0
```
This will create run logs/model checkpoints under `gs://<bucket_name>/pretrain/runs/<run_name>`.
