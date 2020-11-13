"""Script which loads multiple datasets and prepares them for finetuning"""
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


REQUIRED_COLUMNS = ['id', 'label', 'text']
DATA_DIR = os.path.join('..', 'data')
VOCAB_PATH = os.path.join('..', 'vocabs')

class TextClassificationProcessor(DataProcessor):
    """Processor for arbitrary text classification data"""

    def __init__(self, labels):
        self.labels = labels

    def save_label_mapping(self, data_dir):
        with open(os.path.join(data_dir, 'label_mapping.json'), 'w') as f:
            json.dump(self.labels, f)

    def get_examples(self, data_dir, _type):
        f_path = os.path.join(data_dir, f'{_type}.tsv')
        lines = self._read_tsv(f_path)
        return self._create_examples(lines, _type)

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, 'test')

    def get_labels(self):
        return self.labels

    @staticmethod
    def get_processor_name():
        return 'text-classification'

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = f'{set_type}-{i}'
            text_a = tokenization.convert_to_unicode(line[REQUIRED_COLUMNS.index('text')])
            if set_type == 'test':
                label = '0'
            else:
                label = tokenization.convert_to_unicode(line[REQUIRED_COLUMNS.index('label')])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def get_tokenizer(model_class):
    model = PRETRAINED_MODELS[model_class]
    vocab_file = os.path.join(VOCAB_PATH, model['vocab_file'])
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=model['lower_case'])
    return tokenizer

def generate_tfrecords(args, dataset_dir, labels):
    """Generates tfrecords from generated tsv files"""
    processor = TextClassificationProcessor(labels)
    # save label mapping
    processor.save_label_mapping(dataset_dir)
    # get tokenizer
    tokenizer = get_tokenizer(args.model_class)
    processor_text_fn = tokenization.convert_to_unicode
    # generate tfrecords
    input_dir = os.path.join(dataset_dir, 'preprocessed')
    output_dir = os.path.join(dataset_dir, 'tfrecords')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    input_meta_data = generate_tf_record_from_data_file(
        processor,
        input_dir,
        tokenizer,
        train_data_output_path=os.path.join(output_dir, 'train.tfrecords'),
        eval_data_output_path=os.path.join(output_dir, 'dev.tfrecords'),
        max_seq_length=args.max_seq_length)
    with tf.io.gfile.GFile(os.path.join(dataset_dir, 'meta.json'), 'w') as writer:
        writer.write(json.dumps(input_meta_data, indent=4) + '\n')
    logger.info(f'Sucessfully wrote tfrecord files to {output_dir}')

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

def main(args):
    # create run dirs
    run_name = get_run_name(args)
    run_dir = os.path.join(DATA_DIR, 'finetune', run_name)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    # find input data
    originals_dir = os.path.join(DATA_DIR, 'finetune', 'originals')
    if args.finetune_datasets is None or len(args.finetune_datasets) ==  0:
        finetune_datasets = os.listdir(originals_dir)
    else:
        finetune_datasets = args.finetune_datasets
    do_lower_case = PRETRAINED_MODELS[args.model_class]['lower_case']
    for dataset in finetune_datasets:
        logger.info(f'Processing dataset {dataset}...')
        preprocessed_folder = os.path.join(run_dir, dataset, 'preprocessed')
        if not os.path.isdir(preprocessed_folder):
            os.makedirs(preprocessed_folder)
        labels = set()
        for _type in ['train', 'dev']:
            f_name = f'{_type}.tsv'
            logger.info(f'Reading data for for type {_type}...')
            f_path = os.path.join(originals_dir, dataset, f_name)
            if not os.path.isfile(f_path):
                logger.info(f'Could not find file {f_path}. Skipping.')
                continue
            df = pd.read_csv(f_path, usecols=REQUIRED_COLUMNS, sep='\t')
            logger.info('Creating preprocessed files...')
            df.loc[:, 'text'] = df.text.apply(preprocess_bert, args=(args, do_lower_case))
            df.to_csv(os.path.join(preprocessed_folder, f_name), columns=REQUIRED_COLUMNS, header=False, index=False, sep='\t')
            # add labels
            labels.update(df.label.unique().tolist())
        logger.info('Creating tfrecords files...')
        # we sort the labels alphabetically in order to maintain consistent label ids
        labels = sorted(list(labels))
        dataset_dir = os.path.join(run_dir, dataset)
        generate_tfrecords(args, dataset_dir, labels)
        # saving config
    f_path_config = os.path.join(run_dir, 'create_finetune_config.json')
    logger.info(f'Saving config to {f_path_config}')
    save_to_json(vars(args), f_path_config)

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--finetune_datasets', type=str, nargs='+', help='Finetune dataset(s) to process. These correspond to folder names in data/finetune. \
            Data should be located in data/finetune/originals/{finetune_dataset}/[train.tsv/dev.tsv/test.tsv]. By default runs all datasets.')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class')
    parser.add_argument('--run_prefix', help='Prefix to be added to all runs. Useful to identify runs')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--username_filler', default='twitteruser', type=str, help='Username filler')
    parser.add_argument('--url_filler', default='twitterurl', type=str, help='URL filler (ignored when replace_urls option is false)')
    add_bool_arg(parser, 'replace_usernames', default=True, help='Replace usernames with filler')
    add_bool_arg(parser, 'replace_urls', default=True, help='Replace URLs with filler')
    add_bool_arg(parser, 'asciify_emojis', default=True, help='Asciifyi emojis')
    add_bool_arg(parser, 'replace_multiple_usernames', default=True, help='Replace "@user @user" with "2 <username_filler>"')
    add_bool_arg(parser, 'replace_multiple_urls', default=True, help='Replace "http://... http://.." with "2 <url_filler>"')
    add_bool_arg(parser, 'standardize_punctuation', default=True, help='Standardize (asciifyi) special punctuation')
    add_bool_arg(parser, 'remove_unicode_symbols', default=True, help='After preprocessing remove characters which belong to unicode category "So"')
    add_bool_arg(parser, 'remove_accented_characters', default=False, help='Remove accents/asciify everything. Probably not recommended.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
