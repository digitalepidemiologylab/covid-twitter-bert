
# COVID-Twitter-BERT
Pretrained BERT-large language model on Twitter data related to COVID-19 


| v1  | 22.5M tweets (633M tokens) | BERT-large-uncased | en | <img src="images/COVID-Twitter-BERT-medium.png"> |


# Pretrained models
| Version  | Training data | Model | Language | Download |
| -------- | ------------- | ----- | -------- | -------- |
| v1  | 22.5M tweets (633M tokens) | BERT-large-uncased | en | [TF2 Checkpoint](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/checkpoint_submodel/covid-twitter-bert-v1.tar.gz) \| [HuggingFace](https://crowdbreaks-public.s3.eu-central-1.amazonaws.com/models/covid-twitter-bert/v1/huggingface/covid-twitter-bert-v1.tar.gz) |

# Usage

## With Tensorflow 2/Keras/TFHub
We have made our model available through [TFHub](). You can directly use the model by providing the URL to the TF hub module.
```python
import tensorflow as tf
import tensorflow_hub as hub

max_seq_length = 128  # Your choice here.
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

## Use Huggingface transformers
You can use our model with the [transformers](https://github.com/huggingface/transformers) library by huggingface. Note: We couldn't yet validate this model.
```python
from transformers import TFBertForPreTraining, BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

input_path = '/path/to/h5-model-folder'
tokenizer = BertTokenizer.from_pretrained(input_path)
model = TFBertForSequenceClassification.from_pretrained(input_path, num_labels=3)
input_ids = tf.constant(tokenizer.encode("Oh, when will this lockdown ever end?", add_special_tokens=True))[None, :]  # Batch size 1
outputs = model(input_ids)
```

## Use our own scripts
If your goal is to train (finetune) a classifier, you can use the code in this repo. For this you will need to download the checkpoint file.

# Authors
* Martin Müller (martin.muller@epfl.ch)
* Per Egil Kummervold (per@capia.no)
