
# COVID-Twitter-BERT

<img align="right" width="350px" src="images/COVID-Twitter-BERT-medium.png">

A Pretrained BERT-large language model on Twitter data related to COVID-19

COVID-Twitter-BERT (CT-BERT) is a transformer-based model pretrained on a large corpus of Twitter messages on the topic of COVID-19. When used on domain specific datasets our evaluation shows a marginal performane increase of 10–30% compared to the base model.

This repository contains all code used in the paper as well as notebooks to fintetune CT-BERT on your own datasets. 

# Pretrained models
| Version  | Training data | Base model | Language | Download |
| -------- | ------------- | ----- | -------- | -------- |
| v1  | 22.5M tweets (633M tokens) | BERT-large-uncased | en | [TF2 Checkpoint](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/checkpoint_submodel/covid-twitter-bert-v1.tar.gz) \| [HuggingFace](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/huggingface/covid-twitter-bert-v1.tar.gz) |

# Usage
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

## Use our own scripts
Our code can be used for the domain specific pretraining of a transformer model (`run_pretrain.py`) and/or the training of a classifier (`run_finetune.py`).

Our code depends on the official [tensorflow/models](https://github.com/tensorflow/models) implementation of BERT under tensorflow 2.2/Keras. This code is therefore not compatible with TF 1.4 trained models using the [google-research/bert](https://github.com/google-research/bert) repository.

In order to use our code you need to set up:
* A personal bucket
* A Google Cloud VM
* A TPU in the same zone as the VM, version 2.2

If you are a researcher you may [apply for access to TPUs](https://www.tensorflow.org/tfrc) and/or [Google Cloud credits](https://edu.google.com/programs/credits/research/?modal_active=none).

### Install
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

### Finetune
_Some instructions soon to follow_
#### Prepare data
#### Train

### Pretrain
_Some instructions soon to follow_
#### Prepare data
#### Train


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

