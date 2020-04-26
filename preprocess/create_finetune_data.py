"""Script which loads multiple datasets and prepares them for finetuning"""
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread
import re
import unicodedata
import os
import logging
import html
import sys
import functools
import json
import tensorflow as tf
sys.path.append('../tensorflow_models')
sys.path.append('..')
from official.nlp.data.classifier_data_lib import DataProcessor, generate_tf_record_from_data_file, InputExample
from official.nlp.bert import tokenization
from utils.misc import ArgParseDefault

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = Credentials.from_service_account_file('/home/martin/.config/gcloud/cb-tpu-projects-service.json', scopes=scope)
gc = gspread.authorize(credentials)
sheet_handler = gc.open('Twitter Evaluation Datasets')
default_sheets = ['vaccine_sentiment_epfl',
        'maternal_vaccine_stance_lshtm',
        'twitter_sentiment_semeval',
        'covid_worry',
        'covid_category',
        'SST-2']
tsv_columns = ['id', 'label', 'text']
transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
user_handle_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
control_char_regex = re.compile(r'[\r\n\t]+')


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
            text_a = tokenization.convert_to_unicode(line[tsv_columns.index('text')])
            if set_type == 'test':
                label = '0'
            else:
                label = tokenization.convert_to_unicode(line[tsv_columns.index('label')])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def generate_tfrecords(args, data_dir, labels):
    """Generates tfrecords from generated tsv files"""
    processor = TextClassificationProcessor(labels)
    # save label mapping
    processor.save_label_mapping(data_dir)
    # get tokenizer (BERT uses word_piece, Albert sentence_piece)
    if args.tokenizer_type == 'word_piece':
        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        processor_text_fn = tokenization.convert_to_unicode
    elif args.tokenizer_type == 'sentence_piece':
        tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
        processor_text_fn = functools.partial(tokenization.preprocess_text, lower=args.do_lower_case)
    else:
        raise ValueError(f'Unsupported tokenizer type {args.tokenizer_type}')
    # generate tfrecords
    output_dir = os.path.join(data_dir, 'tfrecord')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    input_meta_data = generate_tf_record_from_data_file(
        processor,
        data_dir,
        tokenizer,
        train_data_output_path=os.path.join(output_dir, 'train.tfrecord'),
        eval_data_output_path=os.path.join(output_dir, 'dev.tfrecord'),
        max_seq_length=args.max_seq_length)
    with tf.io.gfile.GFile(os.path.join(output_dir, 'meta.json'), 'w') as writer:
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
    df['a'] = 'a'
    return df

def remove_control_characters(s):
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def clean_text(text):
    """Replace some non-standard characters such as ” or ’ with standard characters. """
    # remove anything non-printable
    text = remove_control_characters(text)
    # escape html (like &amp; -> & or &quot; -> ")
    text = html.unescape(text)
    # standardize quotation marks and apostrophes
    text = text.translate(transl_table)
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def replace_user_handles(text):
    # Replace previously added filler user mentions
    text = text.replace('@twitteruser', '@<user>')
    # Replace any other remaining user mentions
    text = re.sub(user_handle_regex, '@<user>', text)
    # fixes a minor issue of double @ signs
    text = re.sub('@@<user>', '@<user>', text)
    return text

def replace_urls(text):
    # Replace previously added filler URL
    text = text.replace('http://anonymisedurl.com', '<url>')
    # Replace any other remaining urls
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', text)

def clean_data(df):
    """Replaces user mentions & standardize text"""
    df.loc[:, 'text'] = df.text.apply(clean_text)
    df.loc[:, 'text'] = df.text.apply(replace_user_handles)
    df.loc[:, 'text'] = df.text.apply(replace_urls)
    return df

def main(args):
    output_dir = os.path.join('..', 'output', 'finetune')
    for s in args.sheets:
        logger.info(f'Reading sheet {s}...')
        df = read_data(s)
        # logger.info('Cleaning data...')
        df = clean_data(df)
        # # write train, dev and test set
        f_path_folder = os.path.join(output_dir, s)
        for _type in ['train', 'dev', 'test']:
            _df = df[df['dataset'] == _type]
            if len(_df) == 0:
                logger.warning(f'Type {_type} has no records. Skipping.')
                continue
            _df = _df.sample(frac=1)
            if not os.path.isdir(f_path_folder):
                os.makedirs(f_path_folder)
            f_path = os.path.join(f_path_folder, f'{_type}.tsv')
            logging.info(f'Writing {len(_df):,} examples to cleaned finetune data {f_path}')
            _df.to_csv(f_path, columns=tsv_columns, header=False, index=False, sep='\t')
        # generate tfrecords files
        logging.info(f'Generating tfrecord files...')
        # we sort the labels alphabetically in order to maintain consistent label ids
        labels = sorted(df.label.unique())
        generate_tfrecords(args, f_path_folder, labels)

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--sheets', default=default_sheets, choices=default_sheets, nargs='+', help='Parse sheets')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--do_lower_case', default=True, type=bool, help='Use lower case')
    parser.add_argument('--vocab_file', default='../vocabs/bert-large-uncased-vocab.txt', type=str, help='Use lower case')
    parser.add_argument('--tokenizer_type', default='word_piece', choices=['word_piece', 'sentence_piece'], type=str, help='BERT uses word_piece, Albert uses sentence_piece')
    parser.add_argument('--sp_tokenizer_path', type=str, help='The path to the model used by sentence piece tokenizer.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
