"""Script to generate tfrecord data for inference"""
import pandas as pd
import os
import datetime
import logging
import sys
import json
import glob
import tensorflow as tf
sys.path.append('../tensorflow_models')
sys.path.append('..')
from official.nlp.data.classifier_data_lib import DataProcessor, generate_tf_record_from_data_file, InputExample
from official.nlp.bert import tokenization
from utils.preprocess import preprocess_bert
from utils.misc import ArgParseDefault, add_bool_arg, save_to_json
from config import PRETRAINED_MODELS
import collections
from tqdm import tqdm
import multiprocessing
import joblib
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


REQUIRED_COLUMNS = ['id', 'label', 'text']
DATA_DIR = os.path.join('..', 'data')
VOCAB_PATH = os.path.join('..', 'vocabs')

def get_tokenizer(model_class):
    model = PRETRAINED_MODELS[model_class]
    vocab_file = os.path.join(VOCAB_PATH, model['vocab_file'])
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=model['lower_case'])
    return tokenizer

def create_example(text, tokenizer, max_seq_length):
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
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
    features = collections.OrderedDict()
    features["input_word_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["input_type_ids"] = create_int_feature(segment_ids)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example

def read_data(sheet_name):
    # Read the vaccine_sentiment_epfl
    worksheet = sheet_handler.worksheet(sheet_name)
    rows = worksheet.get_all_values()
    # Get it into pandas
    df = pd.DataFrame.from_records(rows)
    df.columns = df.iloc[0]
    df  = df.reindex(df.index.drop(0))
    return df

def get_run_name(args):
    # Use timestamp to generate a unique run name
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    if args.run_prefix:
        run_name = f'run_{ts}_{args.run_prefix}'
    else:
        run_name = f'run_{ts}'
    return run_name

def process(f_name, tfrecord_folder, preprocessed_folder, args):
    # tokenizer
    do_lower_case = PRETRAINED_MODELS[args.model_class]['lower_case']
    tokenizer = get_tokenizer(args.model_class)
    # fnames
    f_out_name = os.path.basename(f_name).split('.')[0] + '.tfrecord'
    f_out_tfrecord = os.path.join(tfrecord_folder, f_out_name)
    tfrecord_writer = tf.io.TFRecordWriter(f_out_tfrecord)
    num_lines = sum(1 for _ in tf.io.gfile.GFile(os.path.join(f_name), 'r'))
    f_out_preprocessed = os.path.join(preprocessed_folder, os.path.basename(f_name))
    with tf.io.gfile.GFile(os.path.join(f_name), 'r') as reader:
        for line in tqdm(reader, total=num_lines):
            line_processed = preprocess_bert(line, args, do_lower_case=do_lower_case)
            if args.write_preprocessed_file:
                with tf.io.gfile.GFile(f_out_preprocessed, 'a') as writer:
                    writer.write(line_processed + '\n')
            example = create_example(line_processed, tokenizer, args.max_seq_length)
            tfrecord_writer.write(example.SerializeToString())

def main(args):
    # create run dirs
    run_name = get_run_name(args)
    run_dir = os.path.join(DATA_DIR, 'predict_data', run_name)
    preprocessed_folder = os.path.join(run_dir, 'preprocessed')
    tfrecord_folder = os.path.join(run_dir, 'tfrecord')
    for _dir in [preprocessed_folder, tfrecord_folder]:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
    # set up parallel
    if args.run_in_parallel:
        num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    else:
        num_cpus = 1
    logger.info(f'Running with {num_cpus} CPUs...')
    t_s = time.time()
    parallel = joblib.Parallel(n_jobs=num_cpus)
    process_fn_delayed = joblib.delayed(process)
    res = parallel((process_fn_delayed(
        f_name,
        tfrecord_folder,
        preprocessed_folder,
        args) for f_name in tqdm(args.input_txt_files, desc='Processing input files')))
    # save config
    f_config = os.path.join(run_dir, 'create_predict_config.json')
    logger.info(f'Saving config to {f_config}')
    data = dict(vars(args))
    save_to_json(data, f_config)

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--input_txt_files', type=str, nargs='+', help='Input txt files to process.')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class')
    parser.add_argument('--run_prefix', help='Run prefix')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--username_filler', default='twitteruser', type=str, help='Username filler')
    parser.add_argument('--url_filler', default='twitterurl', type=str, help='URL filler (ignored when replace_urls option is false)')
    add_bool_arg(parser, 'replace_usernames', default=True, help='Replace usernames with filler')
    add_bool_arg(parser, 'replace_urls', default=True, help='Replace URLs with filler')
    add_bool_arg(parser, 'asciify_emojis', default=True, help='Asciifyi emojis')
    add_bool_arg(parser, 'replace_multiple_usernames', default=True, help='Replace "@user @user" with "2 <username_filler>"')
    add_bool_arg(parser, 'replace_multiple_urls', default=True, help='Replace "http://... http://.." with "2 <url_filler>"')
    add_bool_arg(parser, 'remove_unicode_symbols', default=True, help='After preprocessing remove characters which belong to unicode category "So"')
    add_bool_arg(parser, 'remove_accented_characters', default=False, help='Remove accents/asciify everything. Probably not recommended.')
    add_bool_arg(parser, 'write_preprocessed_file', default=True, help='Write preprocess output file')
    add_bool_arg(parser, 'run_in_parallel', default=True, help='Run script in parallel')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
