import sys
sys.path.append('..')
sys.path.append('../tensorflow_models')
from utils.misc import ArgParseDefault, add_bool_arg
from utils.preprocess import preprocess_bert, segment_sentences
from config import PRETRAINED_MODELS

import glob
import datetime
import logging
import os
from tqdm import tqdm
import joblib
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

DATA_FOLDER = os.path.join('..', 'data')

def get_input_files(input_folder):
    return glob.glob(os.path.join(input_folder, '**', '*.txt'))

def main(args):
    input_files = get_input_files(args.input_data)
    logger.info(f'Found {len(input_files):,} input text files')
    
    # preprocess fn
    preprocess_fn = preprocess_bert
    do_lower_case = PRETRAINED_MODELS[args.model_class]['lower_case']

    # create run dirs
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%s')
    if args.run_prefix:
        run_name = f'run_{args.run_prefix}_{ts}'
    else:
        run_name = f'run_{ts}'
    output_folder = os.path.join(DATA_FOLDER, 'pretrain', run_name, 'preprocessed')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # set up parallel processing
    if args.run_in_parallel:
        num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    else:
        num_cpus = 1
    parallel = joblib.Parallel(n_jobs=num_cpus)
    preprocess_fn_delayed = joblib.delayed(preprocess_file)

    res = parallel((preprocess_fn_delayed(
        input_file,
        preprocess_fn,
        output_folder,
        do_lower_case,
        args) for input_file in tqdm(input_files)))
    num_sentences = sum(r[0] for r in res)
    num_tokens = sum(r[1] for r in res)
    num_tweets = sum(r[2] for r in res)
    num_examples = sum(r[3] for r in res)
    num_examples_single_sentence = sum(r[4] for r in res)
    logger.info(f'Collected a total of {num_sentences:,} sentences, {num_tokens:,} tokens from {num_tweets:,} tweets')
    logger.info(f'Collected a total of {num_examples:,} examples, {num_examples_single_sentence:,} examples only contain a single sentence.')
    logger.info(f'All output files can be found under {output_folder}')

def preprocess_file(input_file, preprocess_fn, output_folder, do_lower_case, args):
    _type = os.path.basename(os.path.dirname(input_file))
    if _type in ['train', 'dev', 'test']:
        output_folder = os.path.join(output_folder, _type)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    input_filename = os.path.basename(input_file).split('.txt')[0]
    output_filepath = os.path.join(output_folder, f'{input_filename}_preprocessed.txt')
    total_num_tokens = 0
    total_num_sentences = 0
    num_examples = 0
    num_examples_single_sentence = 0
    num_lines = sum(1 for _ in open(input_file, 'r'))
    with open(input_file, 'r') as f_in, open(output_filepath, 'w') as f_out:
        for i, line in enumerate(tqdm(f_in, total=num_lines)):
            # preprocess
            text = preprocess_fn(line, args, do_lower_case=do_lower_case)
            # segment sentences
            sentences, num_tokens = segment_sentences(text, args)
            if len(sentences) == 0:
                continue
            output_text = ''
            for sentence in sentences:
                output_text += f'{sentence}\n'
            output_text += f'\n'
            f_out.write(output_text)
            # stats
            total_num_tokens += num_tokens
            total_num_sentences += len(sentences)
            if len(sentences) == 1:
                num_examples_single_sentence += 1
            num_examples += 1
            if i < args.num_logged_samples:
                print('***** Example input *****')
                print(line)
                print(output_text.strip())
                print('***************************')
    logger.info(f'Finished writing file {output_filepath}')
    return total_num_sentences, total_num_tokens, num_lines, num_examples, num_examples_single_sentence


def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--input_data', default='/drives/sde/wuhan_project/preprocess/data/other/pretrain/run_2020_04_26-15-25_1587907545', help='Path to folder with txt files. \
            Folder may contain train/dev/test subfolders. Each txt file contains the text of a single tweet per line.')
    parser.add_argument('--run_prefix', help='Prefix to be added to all runs. Useful to identify runs')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class')
    parser.add_argument('--username_filler', default='twitteruser', type=str, help='Username filler')
    parser.add_argument('--url_filler', default='twitterurl', type=str, help='URL filler (ignored when replace_urls option is false)')
    parser.add_argument('--num_logged_samples', default=10, type=int, help='Log first n samples to output')
    add_bool_arg(parser, 'run_in_parallel', default=True, help='Run script in parallel')
    add_bool_arg(parser, 'replace_usernames', default=True, help='Replace usernames with filler')
    add_bool_arg(parser, 'replace_urls', default=True, help='Replace URLs with filler')
    add_bool_arg(parser, 'asciify_emojis', default=True, help='Asciifyi emojis')
    add_bool_arg(parser, 'replace_multiple_usernames', default=True, help='Replace "@user @user" with "2 <username_filler>"')
    add_bool_arg(parser, 'replace_multiple_urls', default=True, help='Replace "http://... http://.." with "2 <url_filler>"')
    add_bool_arg(parser, 'remove_unicode_symbols', default=True, help='After preprocessing remove characters which belong to unicode category "So"')
    add_bool_arg(parser, 'remove_accented_characters', default=False, help='Remove accents/asciify everything. Probably not recommended.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
