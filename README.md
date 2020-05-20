
# COVID-Twitter-BERT

<img align="right" width="350px" src="images/COVID-Twitter-BERT-medium.png">

COVID-Twitter-BERT (CT-BERT) is a transformer-based model pretrained on a large corpus of Twitter messages on the topic of COVID-19. The model is trained on 22.5M tweets (633M tokens).

When used on domain specific datasets our evaluation shows that this model will get a marginal performance increase of 10–30% compared to the standard BERT-Large-model. Most improvements are shown on COVID-19 related and on Twitter-like messages. 

This repository contains all code and references to models and datasets used in [our paper](https://arxiv.org/pdf/2005.07503.pdf) as well as notebooks to finetune CT-BERT on your own datasets. If you end up using our work, please cite it:
```
Martin Müller, Marcel Salathé, and Per E Kummervold. 
COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter. 
arXiv preprint arXiv:2005.07502, 2020.
```


# Colaboratory
For a demo on how to train a classifier on top of CT-BERT, please take a look at this Collaboratory. It finetunes a model on the SST-2 dataset. It can also easily be modified for finetuning on your own data. 
<p align="center"><a href="https://colab.research.google.com/drive/1cIDAz19ASnQD4OeaYzZo6s2LLzSWLH_7?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


# Load CT-BERT directly
If you are familiar with finetuning Transformer-models, the CT-BERT-model is available both as an downloadable archive, in TFHub and as a module in Huggingface.

| Version  |  Base model | Language | TF2 | Huggingface | TFHub |
| -------- |  ----- | -------- | -------- |------------- |------------- |
| COVID-Twitter-BERT v1  | BERT-large-uncased-WWM | en | [TF2 Checkpoint](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/checkpoint_submodel/covid-twitter-bert-v1.tar.gz) |[Huggingface](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert)| [TFHub](https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/1)|

See below for some short sample code for how this can be integrated. This is provided mainly as a starting point.
<details>
  <summary>Hugginface code example</summary>
  
 ```python
from transformers import (
    TFBertForPreTraining,
    BertTokenizer,
    TFBertForSequenceClassification,
)
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
model = TFBertForSequenceClassification.from_pretrained(
    "digitalepidemiologylab/covid-twitter-bert", num_labels=3
)
input_ids = tf.constant(
    tokenizer.encode("Oh, when will this lockdown ever end?", add_special_tokens=True)
)
model(input_ids[None, :])  # Batch size 1
# (<tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[ 0.17217427, -0.31084645, -0.47540542]], dtype=float32)>,)
```

</details>
<details>
  <summary>TFHub code example</summary>
  
 ```python
#Load the saved model directly
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

#Create a classifier model
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

#Load the tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
```

</details>


# Finetune CT-BERT
The script `run_finetune.py` can be used for training a classifier. This code depends on the official [tensorflow/models](https://github.com/tensorflow/models) implementation of BERT under tensorflow 2.2/Keras.

In order to use our code you need to set up:
* A Google Cloud bucket
* A Google Cloud VM running Tensorflow 2.2
* A TPU in the same zone as the VM also running Tensorflow 2.2

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

## Prepare the data
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

## Start finetuning
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

Run the script 'sync_bucket_data.py' from your local computer to download all the training logs to `data/<bucket_name>/covid-bert/finetune/<run_names>`

```bash
python sync_bucket_data.py --bucket_name <bucket_name>
```

# Datasets
In our preliminary study we have evaluated our model on five different classification datasets

<img align="center" src="images/COVID-Twitter-BERT-graph.jpeg">

| Dataset name  | Num classes | Reference |
| ------------- | ----------- | ----------|
| COVID Category (CC)  | 2 | [Read more](datasets/covid_category) |
| Vaccine Sentiment (VS)  | 3 | [See :arrow_right:](https://github.com/digitalepidemiologylab/crowdbreaks-paper) |
| Maternal vaccine Sentiment (MVS)  | 4 | [not yet public] |
| Stanford Sentiment Treebank 2 (SST-2) | 2 | [See :arrow_right:](https://gluebenchmark.com/tasks) | 
| Twitter Sentiment SemEval (SE) | 3 | [See :arrow_right:](http://alt.qcri.org/semeval2016/task4/index.php?id=data-and-tools) | 
If you end up using these datasets, please make sure to properly cite them.



# Pretrain
A documentation of how we created CT-BERT can be found [here](README_pretrain.md).

# How do I cite COVID-Twitter-BERT?
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
Martin Müller, Marcel Salathé, and Per E Kummervold. 
COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter. 
arXiv preprint arXiv:2005.07502, 2020.
```

# Acknowledgement
* Thanks to Aksel Kummervold for creating the COVID-Twitter-Bert logo

# Authors
* Martin Müller (martin.muller@epfl.ch)
* Per Egil Kummervold (per@capia.no)

