import sys
sys.path.append('../tensorflow_models')
sys.path.append('..')
from utils.misc import ArgParseDefault, add_bool_arg, save_to_json
from config import PRETRAINED_MODELS
from pretrain_helpers import create_instances_from_document, write_instance_to_example_files
from official.nlp.bert import tokenization
import random
import logging
import os
import glob
import joblib
import multiprocessing
from tqdm import tqdm, trange
from collections import defaultdict
import time
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

VOCAB_PATH = os.path.join('..', 'vocabs')
DATA_DIR = os.path.join('..', 'data')

def get_tokenizer(model_class):
    model = PRETRAINED_MODELS[model_class]
    vocab_file = os.path.join(VOCAB_PATH, model['vocab_file'])
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=model['lower_case'])
    return tokenizer

def get_input_files(run_folder):
    input_files = glob.glob(os.path.join(run_folder, 'preprocessed', '**', '*.txt'))
    if len(input_files) == 0:
        raise ValueError(f'No txt files found in folder {run_folder}/preprocessed')
    return input_files

def main(args):
    rng = random.Random(args.random_seed)
    run_folder = os.path.join(DATA_DIR, 'pretrain', args.run_name)
    input_files = get_input_files(run_folder)

    logger.info('Processing the following {len(input_files):,} input files:')
    for input_file in input_files:
        logger.info(f'{input_file}')

    logger.info(f'Setting up tokenizer for model class {args.model_class}')
    tokenizer = get_tokenizer(args.model_class)

    if args.run_in_parallel:
        num_cpus = max(min(multiprocessing.cpu_count() - 1, args.max_num_cpus), 1)
    else:
        num_cpus = 1
    logger.info(f'Running with {num_cpus} CPUs...')
    t_s = time.time()
    parallel = joblib.Parallel(n_jobs=num_cpus)
    process_fn_delayed = joblib.delayed(process)
    res = parallel((process_fn_delayed(
        input_file,
        tokenizer,
        rng,
        args) for input_file in tqdm(input_files, desc='Processing input files')))
    t_e = time.time()
    time_taken_min = (t_e - t_s)/60
    logger.info(f'Finished after {time_taken_min:.1f} min')
    counts = {}
    for _r in res:
        _type = _r[2]
        if _type not in counts:
            counts[_type] = defaultdict(int)
        counts[_type]['num_documents'] += _r[0]
        counts[_type]['num_instances'] += _r[1]
    for _type, c in counts.items():
        num_instances = c['num_instances']
        num_documents = c['num_documents']
        logger.info(f'Type {_type}: Generated a total of {num_instances:,} training examples from {num_documents:,} documents')
    f_config = os.path.join(run_folder, 'create_pretrain_config.json')
    logger.info(f'Saving config to {f_config}')
    data = {
            'counts': counts,
            'time_taken_min': time_taken_min,
            **vars(args)
            }
    save_to_json(data, f_config)

def process(input_file, tokenizer, rng, args):
    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    logger = logging.getLogger(__name__)
    # read & tokenize docs
    all_documents = [[]]
    logger.info('Tokenizing documents...')
    num_logged_examples = 0
    with tf.io.gfile.GFile(input_file, 'rb') as reader:
        num_lines = sum(1 for _ in reader.readline())
    pbar = tqdm(total=num_lines, desc='Tokenization')
    with tf.io.gfile.GFile(input_file, 'rb') as reader:
        i = 0
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            # Empty lines are used as document delimiters
            if not line:
                all_documents.append([])
            tokens = tokenizer.tokenize(line)
            if tokens:
                all_documents[-1].append(tokens)
                if num_logged_examples < args.num_logged_samples:
                    print('**** Tokenization example ****')
                    print(line)
                    print(tokens)
                    print('****')
                    num_logged_examples += 1
            i += 1
            pbar.update(i)
    # shuffle
    logger.info('Shuffling documents...')
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    num_documents = len(all_documents)
    logger.info(f'Tokenized a total of {num_documents:,} documents')
    # create instances
    logger.info('Creating instances...')
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    do_whole_word_masking = PRETRAINED_MODELS[args.model_class]['do_whole_word_masking']
    for _ in range(args.dupe_factor):
        for document_index in trange(len(all_documents), desc='Generating training instances'):
            instances.extend(create_instances_from_document(
                    all_documents,
                    document_index,
                    args.max_seq_length,
                    args.short_seq_prob,
                    args.masked_lm_prob,
                    args.max_predictions_per_seq,
                    vocab_words,
                    rng,
                    do_whole_word_masking))
    all_documents = None # free memory
    num_instances = len(instances)
    logger.info(f'Collected a total of {num_instances:,} training instances')
    logger.info('Shuffling training instances...')
    rng.shuffle(instances)
    # write tf records file
    _type = os.path.basename(os.path.dirname(input_file))
    if _type in ['train', 'dev', 'test']:
        output_folder = os.path.join(DATA_DIR, 'pretrain', args.run_name, 'tfrecords', _type)
    else:
        _type = 'default'
        output_folder = os.path.join(DATA_DIR, 'pretrain', args.run_name, 'tfrecords')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    input_file_name = os.path.basename(input_file)
    output_file = os.path.join(output_folder, f'{input_file_name}.tfrecords')
    logger.info(f'Writing to {output_file}...')
    write_instance_to_example_files(
            instances, tokenizer,
            args.max_seq_length,
            args.max_predictions_per_seq,
            [output_file],
            args.gzipped)
    return num_documents, num_instances, _type

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--run_name', required=True, help='Run name to create tf record files for. Run folder has to be located under \
            data/pretrain/{run_name}/preprocessed/ and must contain one or multiple txt files. May also contain train and dev subfolders with txt files.')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    parser.add_argument('--dupe_factor', default=10, type=int, help='Number of times to duplicate the input data (with different masks).')
    parser.add_argument('--gzipped', action='store_true', default=False, help='Create gzipped tfrecords files')
    parser.add_argument('--short_seq_prob', default=0.1, type=float, help='Probability of creating sequences which are shorter than the maximum length.')
    parser.add_argument('--max_predictions_per_seq', default=14, type=int, help='Maximum number of masked LM predictions per sequence.')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed')
    parser.add_argument('--masked_lm_prob', default=0.15, type=float, help='Masked LM probabibility')
    parser.add_argument('--num_logged_samples', default=10, type=int, help='Log first n samples to output')
    parser.add_argument('--max_num_cpus', default=10, type=int, help='Adapt this number based on the available memory/size of input files. \
            This code was tested on a machine with a lot of memory (250GB). Decrease this number if you run into memory issues.')
    add_bool_arg(parser, 'run_in_parallel', default=True, help='Run script in parallel')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
