import sys
# import from official repo
sys.path.append('tensorflow_models')
from official.utils.misc import distribution_utils
from official.nlp.bert import input_pipeline
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import model_training_utils
from official.nlp import optimization
from official.utils.misc import keras_utils

import os
import datetime
import argparse
import logging
import tqdm
import json
import tensorflow as tf
from utils.misc import ArgParseDefault

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# remove duplicate logger (not sure why this is happening, possibly an issue with the imports in tf/tf_hub)
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

PRETRAINED_MODELS = {
    'bert_large_uncased': 'pretrained_models/bert/keras_bert/uncased_L-24_H-1024_A-16'
}

def get_model_config(config_path):
    config = bert_configs.BertConfig.from_json_file(config_path)
    return config

def get_dataset_fn(args):
    """Returns input dataset from input file string."""
    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed BERT pretraining."""
        input_data = [f'gs://{args.bucket_name}/{args.project_name}/pretrain/pretrain_data/{args.pretrain_data}/pretrain_anonymized_bert_train_???.txt.tfrecords']
        batch_size = ctx.get_per_replica_batch_size(args.train_batch_size)
        train_dataset = input_pipeline.create_pretrain_dataset(
            input_data,
            args.max_seq_length,
            args.max_predictions_per_seq,
            batch_size,
            is_training=True,
            input_pipeline_context=ctx)
        return train_dataset
    return _dataset_fn

def set_mixed_precision_policy(args):
    """Sets mix precision policy."""
    if args.dtype == 'fp16':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale=loss_scale)
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif args.dtype == 'bf16':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif args.dtype == 'fp32':
        tf.keras.mixed_precision.experimental.set_policy('float32')
    else:
        raise ValueError(f'Unknown dtype {args.dtype}')

def configure_optimizer(optimizer, use_float16=False, use_graph_rewrite=False, loss_scale='dynamic'):
    """Configures optimizer object with performance options."""
    if use_float16:
        # Wraps optimizer with a LossScaleOptimizer. This is done automatically in compile() with the
        # "mixed_float16" policy, but since we do not call compile(), we must wrap the optimizer manually.
        optimizer = (tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale=loss_scale))
    if use_graph_rewrite:
        # Note: the model dtype must be 'float32', which will ensure
        # tf.ckeras.mixed_precision and tf.train.experimental.enable_mixed_precision_graph_rewrite do not double up.
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    return optimizer

def get_loss_fn():
    """Returns loss function for BERT pretraining."""
    def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
        return tf.reduce_mean(losses)
    return _bert_pretrain_loss_fn

def run(args, strategy):
    """Pretrains model using TF2. Adapted from the tensorflow/models Github"""
    # CONFIG
    # Use timestamp to generate a unique run name
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%s')
    if args.run_prefix:
        run_name = f'run_{args.run_prefix}_{ts}'
    else:
        run_name = f'run_{ts}'
    logger.info(f'*** Starting run {run_name} ***')
    output_dir = f'gs://{args.bucket_name}/{args.project_name}/pretrain/runs/{run_name}'

    # pretrained model path
    model_key = f'{args.model_class}_{args.model_type}'
    try:
        pretrained_model_path = PRETRAINED_MODELS[model_key]
    except KeyError:
        raise ValueError(f'Could not find a pretrained model matching the model class {args.model_class} and {args.model_type}')
    pretrained_model_config_path = f'gs://{args.bucket_name}/{pretrained_model_path}/bert_config.json'
    pretrained_model_checkpoint_path = f'gs://{args.bucket_name}/{pretrained_model_path}/bert_model.ckpt'

    # load model config based on model_class and model_type
    model_config = get_model_config(pretrained_model_config_path)

    # input data function
    train_input_fn = get_dataset_fn(args)

    # model_fn
    def _get_pretrained_model(end_lr=0.0):
        """Gets a pretraining model."""
        pretrain_model, core_model = bert_models.pretrain_model(model_config, args.max_seq_length, args.max_predictions_per_seq)
        optimizer = optimization.create_optimizer(
                args.learning_rate,
                args.num_steps_per_epoch * args.num_epochs,
                args.warmup_steps,
                args.end_lr,
                args.optimizer_type)
        pretrain_model.optimizer = configure_optimizer(optimizer, use_float16=args.dtype == 'fp16', use_graph_rewrite=False)
        return pretrain_model, core_model

    # custom callbacks
    summary_dir = os.path.join(output_dir, 'summaries')
    time_history_callback = keras_utils.TimeHistory(
        batch_size=args.train_batch_size,
        log_steps=args.time_history_log_steps,
        logdir=summary_dir)
    custom_callbacks = [time_history_callback]

    # run training loop
    model_training_utils.run_customized_training_loop(
        strategy=strategy,
        model_fn=_get_pretrained_model,
        loss_fn=get_loss_fn(),
        scale_loss=True,
        model_dir=output_dir,
        train_input_fn=train_input_fn,
        steps_per_epoch=args.num_steps_per_epoch,
        steps_per_loop=args.steps_per_loop,
        epochs=args.num_epochs,
        eval_input_fn=None,
        eval_steps=None,
        metric_fn=None,
        init_checkpoint=pretrained_model_checkpoint_path,
        custom_callbacks=custom_callbacks,
        run_eagerly=False,
        sub_model_export_name='pretrained/bert_model',
        explicit_allreduce=False,
        pre_allreduce_callbacks=None,
        post_allreduce_callbacks=None)

def main(args):
    # Get distribution strategy
    if args.not_use_tpu:
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored', num_gpus=args.num_gpus)
    else:
        logger.info(f'Intializing TPU on address {args.tpu_ip}...')
        tpu_address = f'grpc://{args.tpu_ip}:8470'
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='tpu', tpu_address=tpu_address, num_gpus=args.num_gpus)
    # set mixed precision
    set_mixed_precision_policy(args)
    run(args, strategy)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', help='Prefix to be added to all runs. Useful to group runs')
    parser.add_argument('--pretrain_data', default='v1', type=str, help='Folder which contains pretrain data. Should be located under gs://{bucket_name}/{project_name}/pretrain/pretrain_data/')
    parser.add_argument('--bucket_name', default='cb-tpu-projects', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--tpu_ip', default='10.74.219.210', help='IP-address of the TPU')
    parser.add_argument('--not_use_tpu', action='store_true', default=False, help='Do not use TPUs')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--optimizer_type', default='adamw', choices=['adamw', 'lamb'], type=str, help='Optimizer')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Training batch size')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs')
    parser.add_argument('--num_steps_per_epoch', default=1000, type=int, help='Number of steps per epoch')
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Warmup steps')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--end_lr', default=0, type=float, help='Final learning rate')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length. Sequences longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--max_predictions_per_seq', default=14, type=int, help='Maximum predictions per sequence_output.')
    parser.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16', 'fp16'], type=str, help='Data type')
    parser.add_argument('--model_class', default='bert', type=str, choices=['bert'], help='Model class')
    parser.add_argument('--model_type', default='large_uncased', choices=['large_uncased', 'base_uncased'], type=str, help='Model class to pretraining')
    parser.add_argument('--steps_per_loop', default=10, type=int, help='Steps per loop')
    parser.add_argument('--time_history_log_steps', default=1000, type=int, help='Frequency with which to log timing information with TimeHistory.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
