import sys
# import from official repo
sys.path.append('../tensorflow_models')
sys.path.append('..')
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import tokenization
from config import PRETRAINED_MODELS
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import logging
import argparse
import tensorflow as tf
import os
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

VOCAB_PATH = '../vocabs'

def create_example(text, tokenizer, max_seq_length):
    tokens = ['[CLS]']
    input_tokenized = tokenizer.tokenize(text)
    if len(input_tokenized) + 2 > max_seq_length:
        # truncate
        input_tokenized = input_tokenized[:(max_seq_length + 2)]
    tokens.extend(input_tokenized)
    tokens.append('[SEP]')
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    num_tokens = len(input_ids)
    input_mask = num_tokens * [1]
    # pad
    input_ids += (max_seq_length - num_tokens) * [0]
    input_mask += (max_seq_length - num_tokens) * [0]
    segment_ids = max_seq_length * [0]
    return tf.constant(input_ids, dtype=tf.int32), tf.constant(input_mask, dtype=tf.int32), tf.constant(segment_ids, dtype=tf.int32)

def generate_single_example(text, tokenizer, max_seq_length):
    example = create_example(text, tokenizer, max_seq_length)
    example_features = {
        'input_word_ids': example[0][None, :],
        'input_mask': example[1][None, :],
        'input_type_ids': example[2][None, :]
    }
    return example_features

def get_tokenizer(model_class):
    model = PRETRAINED_MODELS[model_class]
    vocab_file = os.path.join(VOCAB_PATH, model['vocab_file'])
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=model['lower_case'])
    return tokenizer

def load_tf_model(tf_config_path, num_labels=3, max_seq_length=96):
    config = bert_configs.BertConfig.from_json_file(tf_config_path)
    classifier_model, _ = bert_models.classifier_model(
            config,
            num_labels,
            max_seq_length,
            hub_module_url=None,
            hub_module_trainable=False)
    return classifier_model

def hugg_generate_single_example(input_text, tokenizer):
    inputs = tokenizer(input_text, return_tensors='pt')
    return inputs

def main(args):
    # configs
    bert_config = BertConfig.from_pretrained(args.path_to_huggingface)

    # tokenizer
    tf_tokenizer = get_tokenizer(args.model_class)
    hugg_tokenizer = BertTokenizer.from_pretrained(args.model_class_huggingface)

    # models
    tf_config_path = os.path.join(args.path_to_tf, 'bert_config.json')
    tf_model = load_tf_model(tf_config_path, num_labels=bert_config.num_labels, max_seq_length=bert_config.max_seq_length)
    checkpoint_path = os.path.join(args.path_to_tf, 'checkpoint')
    logger.info(f'Restore run checkpoint {checkpoint_path}...')
    tf_model.load_weights(checkpoint_path)
    hugg_model = BertForSequenceClassification.from_pretrained(args.path_to_huggingface)

    # generate examples and run inference
    tf_example = generate_single_example(args.input_text, tf_tokenizer, bert_config.max_seq_length)
    hugg_example = hugg_generate_single_example(args.input_text, hugg_tokenizer)

    tf_preds = tf_model.predict(tf_example)
    hugg_preds = hugg_model(**hugg_example)

    print('Huggingface predictions:')
    print(hugg_preds)

    print('TF predictions:')
    print(tf_preds)

def parse_args():
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_huggingface', type=str, required=True, help='Path to folder containing pytorch model/config.json for huggingface')
    parser.add_argument('--path_to_tf', type=str, required=True, help='Path to folder containing tf checkpoint and bert_config.json')
    parser.add_argument('--input_text', default='this is some example test', required=False, help='Test example')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    parser.add_argument('--model_class_huggingface', default='bert-large-uncased', help='Model class to use for huggingface tokenizer')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
