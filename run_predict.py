USAGE_DESCRIPTION = """
Run prediction by loading a fine-tuned model
"""

import sys
# import from official repo
sys.path.append('tensorflow_models')
from official.utils.misc import distribution_utils
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import tokenization

import os
import datetime
import time
import argparse
import logging
from tqdm import tqdm
import json
import tensorflow as tf
from utils.misc import ArgParseDefault, add_bool_arg, save_to_json
import utils.optimizer
from config import PRETRAINED_MODELS
import collections
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)
VOCAB_PATH = 'vocabs'

# remove duplicate logger (not sure why this is happening, possibly an issue with the imports)
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

def get_model(args, model_config, num_labels, max_seq_length, is_hub_module=False):
    classifier_model, _ = bert_models.classifier_model(
            model_config,
            num_labels,
            max_seq_length,
            hub_module_url=None,
            hub_module_trainable=False)
    return classifier_model

def get_model_config_path(args):
    try:
        config_path = PRETRAINED_MODELS[args.model_class]['config']
    except KeyError:
        raise ValueError(f'Could not find a pretrained model matching the model class {args.model_class}')
    return os.path.join('configs', config_path)

def get_model_config(config_path):
    config = bert_configs.BertConfig.from_json_file(config_path)
    return config

def read_run_log(run_dir):
    with tf.io.gfile.GFile(os.path.join(run_dir, 'run_logs.json'), 'rb') as reader:
        run_log = json.loads(reader.read().decode('utf-8'))
    return run_log

def get_tokenizer(model_class):
    model = PRETRAINED_MODELS[model_class]
    vocab_file = os.path.join(VOCAB_PATH, model['vocab_file'])
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=model['lower_case'])
    return tokenizer

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

def format_prediction(preds, label_mapping, label_name):
    preds = tf.nn.softmax(preds, axis=1)
    formatted_preds = []
    for pred in preds.numpy():
        # convert to Python types and sort
        pred = {label: float(probability) for label, probability in zip(label_mapping.values(), pred)}
        pred = {k: v for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)}
        formatted_preds.append({label_name: list(pred.keys())[0], f'{label_name}_probabilities': pred})
    return formatted_preds

def generate_single_example(text, tokenizer, max_seq_length):
    example = create_example(text, tokenizer, max_seq_length)
    example_features = {
        'input_word_ids': example[0][None, :],
        'input_mask': example[1][None, :],
        'input_type_ids': example[2][None, :]
    }
    return example_features

def generate_examples_from_txt_file(input_file, tokenizer, max_seq_length, batch_size):
    dataset = tf.data.TextLineDataset(input_file)
    dataset = dataset.batch(batch_size)
    for batch in dataset:
        batch = batch.numpy()
        batch = [create_example(t, tokenizer, max_seq_length) for t in batch]
        yield {
                'input_word_ids': tf.stack([b[0] for b in batch], axis=0),
                'input_mask': tf.stack([b[1] for b in batch], axis=0),
                'input_type_ids': tf.stack([b[2] for b in batch], axis=0)
                }

def run(args):
    # start time
    s_time = time.time()
    # paths
    run_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/runs/{args.run_name}'
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    output_folder = os.path.join('data', 'predictions', f'predictions_{ts}')
    predictions_output_folder = os.path.join('data', 'predictions', f'predictions_{ts}', 'predictions')
    if not os.path.isdir(predictions_output_folder):
        os.makedirs(predictions_output_folder)
    # read configs
    logger.info(f'Reading run configs...')
    run_log = read_run_log(run_dir)
    pretrained_model_config_path = get_model_config_path(args)
    model_config = get_model_config(pretrained_model_config_path)
    max_seq_length = run_log['max_seq_length']
    label_mapping = run_log['label_mapping']
    num_labels = len(label_mapping)
    # load tokenizer
    logger.info(f'Loading tokenizer...')
    tokenizer = get_tokenizer(args.model_class)
    # load model
    logger.info(f'Loading model...')
    model = get_model(args, model_config, num_labels, max_seq_length)
    # restore fine-tuned run
    checkpoint_path = os.path.join(run_dir, 'checkpoint')
    logger.info(f'Restore run checkpoint {checkpoint_path}...')
    # load weights (expect partial state because we don't want need the optimizer state)
    model.load_weights(checkpoint_path)
    logger.info(f'... successfully restored checkpoint')
    # predict
    num_predictions = 0
    predictions = []
    if args.input_text:
        example = generate_single_example(args.input_text, tokenizer, max_seq_length)
        preds = model.predict(example)
        preds = format_prediction(preds, label_mapping, args.label_name)
        print(json.dumps(preds, indent=4))
    elif args.interactive_mode:
        while True:
            text = input('Type text to predict. Quit by typing "q".\n>>> ')
            if text.lower() == 'q':
                break
            example = generate_single_example(text, tokenizer, max_seq_length)
            preds = model.predict(example)
            preds = format_prediction(preds, label_mapping, args.label_name)
            print(json.dumps(preds, indent=4))
        return
    elif args.input_txt_files:
        s_time_predict = time.time()
        for input_file in args.input_txt_files:
            num_lines = sum(1 for _ in open(input_file, 'r'))
            num_batches = int(num_lines/args.eval_batch_size) + 1
            f_out_name = os.path.basename(input_file).split('.')[-2]
            f_out = os.path.join(predictions_output_folder, f'{f_out_name}.jsonl')
            logger.info(f'Predicting file {input_file}...')
            for batch in tqdm(generate_examples_from_txt_file(input_file, tokenizer, max_seq_length, args.eval_batch_size), total=num_batches, unit='batches'):
                preds = model.predict(batch)
                preds = format_prediction(preds, label_mapping, args.label_name)
                num_predictions += len(preds)
                with open(f_out, 'a') as f:
                    for pred in preds:
                        f.write(json.dumps(pred) + '\n')
        e_time_predict = time.time()
        prediction_time_min = (e_time_predict - s_time_predict)/60
        logger.info(f'Wrote {num_predictions:,} predictions in {prediction_time_min:.1f} min ({num_predictions/prediction_time_min:.1f} predictions per min)')
    elif args.input_tfrecord_files:
        def decode_record(record):
            """Decodes a record to a TensorFlow example."""
            name_to_features = {
                'input_word_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
                'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
                'input_type_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            }
            example = tf.io.parse_single_example(record, name_to_features)
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                    example[name] = t
            return example
        s_time_predict = time.time()
        for input_file in args.input_tfrecord_files:
            dataset = tf.data.TFRecordDataset(input_file)
            dataset = dataset.map(decode_record)
            dataset = dataset.batch(args.eval_batch_size)
            num_records = sum(1 for _ in tf.data.TFRecordDataset(input_file))
            f_out_name = os.path.basename(input_file).split('.')[-2]
            f_out = os.path.join(predictions_output_folder, f'{f_out_name}.jsonl')
            for batch in tqdm(dataset, total=int(num_records/args.eval_batch_size) + 1, unit='batches'):
                preds = model.predict(batch)
                preds = format_prediction(preds, label_mapping, args.label_name)
                num_predictions += len(preds)
                with open(f_out, 'a') as f:
                    for pred in preds:
                        f.write(json.dumps(pred) + '\n')
        e_time_predict = time.time()
        prediction_time_min = (e_time_predict - s_time_predict)/60
        logger.info(f'Wrote {num_predictions:,} predictions in {prediction_time_min:.1f} min ({num_predictions/prediction_time_min:.1f} predictions per min)')
    e_time = time.time()
    total_time_min = (e_time - s_time)/60
    f_config = os.path.join(output_folder, 'predict_config.json')
    logger.info(f'Saving config to {f_config}')
    data = {
            'prediction_time_min': prediction_time_min,
            'total_time_min': total_time_min,
            'num_predictions': num_predictions,
            **vars(args)}
    save_to_json(data, f_config)

def main(args):
    # Get distribution strategy
    if args.use_tpu:
        logger.info(f'Intializing TPU on address {args.tpu_ip}...')
        tpu_address = f'grpc://{args.tpu_ip}:8470'
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='tpu', tpu_address=tpu_address, num_gpus=args.num_gpus)
    else:
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored', num_gpus=args.num_gpus)
    # Run training
    with strategy.scope():
        run(args)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault(usage=USAGE_DESCRIPTION)
    parser.add_argument('--run_name', required=True, help='Finetune run name. The model will be loaded from gs://{bucket_name}/{project_name}/finetune/runs/{run_name}.')
    parser.add_argument('--bucket_name', required=True, help='Bucket name')
    parser.add_argument('--project_name', required=False, default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--input_text', required=False, help='Predict arbitrary input text and print prediction to stdout')
    parser.add_argument('--input_txt_files', nargs='+', required=False, help='Predict text from txt files. One example per line.')
    parser.add_argument('--input_tfrecord_files', nargs='+', required=False, help='Predict text from tfrecord files.')
    parser.add_argument('--tpu_ip', required=False, help='IP-address of the TPU')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--label_name', default='label', type=str, help='Name of label to predicted')
    add_bool_arg(parser, 'interactive_mode', default=False, help='Interactive mode')
    add_bool_arg(parser, 'use_tpu', default=False, help='Use TPU')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
