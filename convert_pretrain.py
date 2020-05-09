import os
import shutil
import sys
import logging
# import from official repo
sys.path.append('tensorflow_models')
import tensorflow as tf
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from utils.misc import ArgParseDefault
from config import PRETRAINED_MODELS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# remove duplicate logger
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

def main(args):
    # paths
    model_location = PRETRAINED_MODELS[args.model_class]['location']
    pretrained_model_path = f'gs://{args.bucket_name}/{model_location}'

    # load model config
    logger.info(f'Loading BERT config')
    pretrained_model_config_path = f'{pretrained_model_path}/bert_config.json'
    model_config = bert_configs.BertConfig.from_json_file(pretrained_model_config_path)

    # prepare keras model
    logger.info(f'Building model')
    _, core_model = bert_models.pretrain_model(model_config, args.max_seq_length, args.max_predictions_per_seq)

    # restore checkpoint
    if args.init_checkpoint:
        # restore model from a specific pretrain checkpoint
        checkpoint_path = f'gs://{args.bucket_name}/{args.project_name}/pretrain/runs/{args.init_checkpoint}'
    else:
        # restore from default model
        checkpoint_path = os.path.join(pretrained_model_path, 'bert_model.ckpt')
    logger.info(f'Restoring checkpoint from {checkpoint_path}')
    checkpoint = tf.train.Checkpoint(model=core_model)
    checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
    logger.info(f'... sucessfully restored model')

    # save keras model in HDF5 format
    if args.init_checkpoint:
        output_folder = args.init_checkpoint.replace('/', '_').replace('.', '_')
    else:
        output_folder = args.model_class
    export_path = os.path.join('data', 'convert', 'output', output_folder)
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    model_export_path = os.path.join(export_path, 'tf_model.h5')
    logger.info(f'Saving keras model to {model_export_path}')
    core_model.save(model_export_path, include_optimizer=False, save_format='h5')
    # save config
    config_export_path = os.path.join(export_path, 'config.json')
    logger.info(f'Saving BERT config to {config_export_path}')
    with open(config_export_path, 'w') as f:
        f.write(model_config.to_json_string())
    # save vocab
    vocab_export_path = os.path.join(export_path, 'vocab.txt')
    logger.info(f'Saving vocab file to {vocab_export_path}')
    vocab_path = os.path.join('vocabs', PRETRAINED_MODELS[args.model_class]['vocab_file'])
    shutil.copy(vocab_path, vocab_export_path)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-projects', help='Bucket name')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    # parser.add_argument('--init_checkpoint', default='run_2020-04-29_11-14-08_711153_wwm_v2/pretrained/bert_model.ckpt-21', help='Path to checkpoint')
    parser.add_argument('--init_checkpoint', default='run_2020-04-29_11-14-08_711153_wwm_v2/ctl_step_150000.ckpt-6', help='Path to checkpoint')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length. Sequences longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--max_predictions_per_seq', default=14, type=int, help='Maximum predictions per sequence_output.')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
