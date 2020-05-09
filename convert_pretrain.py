import os
import shutil
import sys
import logging
# import from official repo
sys.path.append('tensorflow_models')
import tensorflow as tf
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.bert.export_tfhub import export_bert_tfhub
from utils.misc import ArgParseDefault
from config import PRETRAINED_MODELS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# remove duplicate logger
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

def create_bert_model(bert_config):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    bert_config: A `BertConfig` to create the core model.

  Returns:
    A keras model.
  """
  # Adds input layers just as placeholders.
  input_word_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_type_ids")
  transformer_encoder = bert_models.get_transformer_encoder(
      bert_config, sequence_length=None)
  sequence_output, pooled_output = transformer_encoder(
      [input_word_ids, input_mask, input_type_ids])
  # To keep consistent with legacy hub modules, the outputs are
  # "pooled_output" and "sequence_output".
  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output]), transformer_encoder


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
    _, core_model = create_bert_model(model_config)

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

    if args.init_checkpoint:
        output_folder = args.init_checkpoint.replace('/', '_').replace('.', '_')
    else:
        output_folder = args.model_class

    # export to different formats
    if 'huggingface' in args.output:
        logger.info('Exporting version for hugggingface...')
        # save keras model in HDF5 format
        export_path = os.path.join('data', 'convert', 'output', output_folder, 'huggingface')
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
    if 'tf_hub' in args.output:
        logger.info('Exporting version for tfhub...')
        export_path = os.path.join('data', 'convert', 'output', output_folder, 'tf_hub')
        if not os.path.isdir(export_path):
            os.makedirs(export_path)
        vocab_path = os.path.join('vocabs', PRETRAINED_MODELS[args.model_class]['vocab_file'])
        core_model.vocab_file = tf.saved_model.Asset(vocab_path)
        core_model.do_lower_case = tf.Variable(PRETRAINED_MODELS[args.model_class]['lower_case'], trainable=False)
        core_model.save(export_path, include_optimizer=False, save_format="tf")

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-projects', help='Bucket name')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    parser.add_argument('--init_checkpoint', default=None, help='Path to checkpoint')
    # parser.add_argument('--init_checkpoint', default='run_2020-04-29_11-14-08_711153_wwm_v2/pretrained/bert_model.ckpt-21', help='Path to checkpoint')
    # parser.add_argument('--init_checkpoint', default='run_2020-04-29_11-14-08_711153_wwm_v2/ctl_step_150000.ckpt-6', help='Path to checkpoint')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--output', default=['tf_hub', 'huggingface'], choices=['tf_hub', 'huggingface'], nargs='+', help='Name of subfolder in Google bucket')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
