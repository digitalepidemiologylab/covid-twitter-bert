
# COVID-Twitter-BERT

<img align="right" width="350px" src="images/COVID-Twitter-BERT-medium.png">

A Pretrained BERT-large language model on Twitter data related to COVID-19

COVID-Twitter-BERT (CT-BERT) is a transformer-based model pretrained on a large corpus of Twitter messages on the topic of COVID-19. When used on domain specific datasets our evaluation shows a marginal performane increase of 10–30% compared to the base model.

This repository contains all code used in the paper as well as notebooks to fintetune CT-BERT on your own datasets. 

# Pretrained models
| Version  | Training data | Base model | Language | Download |
| -------- | ------------- | ----- | -------- | -------- |
| v1  | 22.5M tweets (633M tokens) | BERT-large-uncased | en | [TF2 Checkpoint](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/checkpoint_submodel/covid-twitter-bert-v1.tar.gz) \| [HuggingFace](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/huggingface/covid-twitter-bert-v1.tar.gz) |

# Quick start
You can either download the above checkpoints or pull the models from [Huggingface](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert) or [TFHub](https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/1) (see examples below). The hosted models include the tokenizer. If you are downloading the checkpoints, make sure to use the official `bert-large-uncased` vocabulary.

## Huggingface transformers
You can create a classifier model with Huggingface by simply providing `digitalepidemiologylab/covid-twitter-bert`
with the `from_pretrained()` syntax:

```python
from transformers import TFBertForPreTraining, BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
model = TFBertForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert', num_labels=3)
input_ids = tf.constant(tokenizer.encode("Oh, when will this lockdown ever end?", add_special_tokens=True))[None, :]  # Batch size 1
model(input_ids)
# (<tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[ 0.17217427, -0.31084645, -0.47540542]], dtype=float32)>,)
```

## TFHub
Make sure to have `tensorflow 2.x` and `tensorflow_hub` installed. You can then instantiate a `KerasLayer` with our TFHub URL `https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/1` and build a classifier model like so: 
```python
import tensorflow as tf
import tensorflow_hub as hub

max_seq_length = 96  # Your choice here.
input_word_ids = tf.keras.layers.Input(
  shape=(max_seq_length,),
  dtype=tf.int32,
  name="input_word_ids")
input_mask = tf.keras.layers.Input(
  shape=(max_seq_length,),
  dtype=tf.int32,
  name="input_mask")
input_type_ids = tf.keras.layers.Input(
  shape=(max_seq_length,),
  dtype=tf.int32,
  name="input_type_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/1", trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
# create classifier model
num_labels = 3
initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2)
output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
output = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer, name='output')(output)
classifier_model = tf.keras.Model(
  inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids}, 
  outputs=output)
```

# Our code
Our code can be used for the domain specific pretraining of a transformer model (`run_pretrain.py`) and/or the training of a classifier (`run_finetune.py`).

Our code depends on the official [tensorflow/models](https://github.com/tensorflow/models) implementation of BERT under tensorflow 2.2/Keras. This code is therefore not compatible with TF 1.4 trained models using the [google-research/bert](https://github.com/google-research/bert) repository.

In order to use our code you need to set up:
* A Google Cloud bucket
* A Google Cloud VM
* A TPU in the same zone as the VM, version 2.2

If you are a researcher you may [apply for access to TPUs](https://www.tensorflow.org/tfrc) and/or [Google Cloud credits](https://edu.google.com/programs/credits/research/?modal_active=none).

## Install
Clone the repository recursively
```bash
git clone https://github.com/digitalepidemiologylab/covid-twitter-bert.git --recursive && cd covid-twitter-bert
```
Our code was developed using `tf-nightly` but we made it backwards compatible to run with tensorflow 2.2. We recommend using Anaconda to manage the Python version:
```bash
conda create -n covid-twitter-bert python=3.8
conda activate covid-twitter-bert
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Finetune
You may finetune CT-BERT on your own classification dataset.
### Prepare data
Split your data into a training set `train.tsv` and a validation set `dev.tsv` with the following format:
```
id      label   text
1224380447930683394     label_a       Example text 1
1224380447930683394     label_a       Example text 2
1220843980633661443     label_b       Example text 3
```
Place these files into the folder `data/finetune/originals/<dataset_name>/(train|dev).tsv` (using your own `dataset_name`).

You can then run
```bash
cd preprocess
python create_finetune_data.py \
  --run_prefix test_run \
  --finetune_datasets <dataset_name> \
  --model_class bert_large_uncased_wwm \
  --max_seq_length 96 \
  --asciify_emojis \
  --username_filler twitteruser \
  --url_filler twitterurl \
  --replace_multiple_usernames \
  --replace_multiple_urls \
  --remove_unicode_symbols
```
This will generate TF record files in `data/finetune/run_2020-05-19_14-14-53_517063_test_run/<dataset_name>/tfrecords`.

You can now upload the data to your bucket:
```bash
cd data
gsutil -m rsync -r finetune/ gs://<bucket_name>/covid-bert/finetune/finetune_data/
```

### Train
You can now finetune CT-BERT on this data using the following command
```bash
RUN_PREFIX=testrun                                  # Name your run
BUCKET_NAME=                                        # Fill in your buckets name here (without the gs:// prefix)
TPU_IP=XX.XX.XXX.X                                  # Fill in your TPUs IP here
FINETUNE_DATASET=<dataset_name>                     # Your dataset name
FINETUNE_DATA=<dataset_run>                         # Fill in dataset run name (e.g. run_2020-05-19_14-14-53_517063_test_run)
MODEL_CLASS=covid-twitter-bert
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=8
LR=2e-5
NUM_EPOCHS=1

python run_finetune.py \
  --run_prefix $RUN_PREFIX \
  --bucket_name $BUCKET_NAME \
  --tpu_ip $TPU_IP \
  --model_class $MODEL_CLASS \
  --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LR
```
Training logs, run configs, etc are then stored to `gs://<bucket_name>/covid-bert/finetune/runs/run_2020-04-29_21-20-52_656110_<run_prefix>/`. Among tensorflow logs you will find a file called `run_logs.json` containing all relevant training information
```
{
    "created_at": "2020-04-29 20:58:23",
    "run_name": "run_2020-04-29_20-51-10_405727_test_run",
    "final_loss": 0.19747886061668396,
    "max_seq_length": 96,
    "num_train_steps": 210,
    "eval_steps": 103,
    "steps_per_epoch": 42,
    "training_time_min": 6.77958079179128,
    "f1_macro": 0.7216383309465823,
    "scores_by_label": {
      ...
    },
    ...
}
```

### Sync results
Syncronize all training logs with your local repository by running
```bash
python sync_bucket_data.py --bucket_name <bucket_name>
```
Which will pull down all logs from that bucket and store them to `data/<bucket_name>/covid-bert/finetune/<run_names>`.

## Pretrain
_Some more information soon to follow_
### Prepare data
### Train


## How do I cite COVID-Twitter-BERT?
You can cite our [preprint](https://arxiv.org/abs/2005.07503):
```bibtex
@article{mueller2020covid,
  title={{COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter}},
  author={M\"uller, Martin and Salath\'e, Marcel and Kummervold, Per E},
  journal={arXiv preprint arXiv:2005.07502},
  year={2020}
}
```
or
```
Martin Müller, Marcel Salathé, and Per E Kummervold. COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter. arXiv preprint arXiv:2005.07502, 2020.
```

## Acknowledgement
* Thanks to Aksel Kummervold for creating the COVID-Twitter-Bert logo

# Authors
* Martin Müller (martin.muller@epfl.ch)
* Per Egil Kummervold (per@capia.no)

