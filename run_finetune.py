import sys
import os
import datetime
import pprint
import uuid
import time
import argparse
import math
import logging
import tqdm
import json
import tensorflow as tf
from utils.misc import ArgParseDefault, append_to_csv
from utils.finetune_helpers import Metrics
import pandas as pd

# import from official repo
sys.path.append('tensorflow_models')
from official.utils.misc import distribution_utils
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp import optimization
from official.modeling import performance
from official.nlp.bert import input_pipeline
from official.utils.misc import keras_utils


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


PRETRAINED_MODELS = {
    'bert_large_uncased': 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16',
    'bert_base_uncased': 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12'
}

def get_model_config(args):
    if args.model_config_path:
        config_path = args.config_path
    else:
        model_key = f'{args.model_class}_{args.model_type}'
        if not model_key in PRETRAINED_MODELS.keys():
            raise ValueError('Could not find a pretrained model matching the model class {args.model_class} and {args.model_type}')
        config_path = os.path.join(PRETRAINED_MODELS[model_key], 'bert_config.json')
    config = bert_configs.BertConfig.from_json_file(config_path)
    return config

def get_input_meta_data(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'tfrecord', 'meta.json'), 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    return input_meta_data

def get_model(args, model_config, steps_per_epoch, warmup_steps, num_labels, max_seq_length):
    # Get classifier and core model (used to initialize from checkpoint)
    classifier_model, core_model = bert_models.classifier_model(
            model_config,
            num_labels,
            max_seq_length,
            hub_module_url=None,
            hub_module_trainable=False)
    # Optimizer
    optimizer = optimization.create_optimizer(
            args.learning_rate,
            steps_per_epoch * args.num_epochs,
            warmup_steps,
            args.optimizer_type)
    # TODO: Support fp16
    classifier_model.optimizer = performance.configure_optimizer(
            optimizer,
            use_float16=False,
            use_graph_rewrite=False)
    return classifier_model, core_model

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training):
  """Gets a closure to create a dataset."""
  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn

def get_loss_fn(num_classes):
    """Gets the classification loss function."""
    def classification_loss_fn(labels, logits):
        """Classification loss."""
        labels = tf.squeeze(labels)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
        return tf.reduce_mean(per_example_loss)
    return classification_loss_fn

def get_label_mapping(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'label_mapping.json'), 'rb') as reader:
        label_mapping = json.loads(reader.read().decode('utf-8'))
    label_mapping = dict(zip(range(len(label_mapping)), label_mapping))
    return label_mapping

def get_metrics():
    return [tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32)]

def train(args, strategy, repeat):
    """Train using the Keras/TF 2.0. Adapted from the tensorflow/models Github"""
    # CONFIG
    # Use timestamp to generate a unique run name
    ts = datetime.datetime.now().strftime('%Y_%m_%d_%s')
    run_name = f'run_{ts}'
    data_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/finetune_data/{args.finetune_data}'
    output_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/runs/{args.finetune_data}/{run_name}'

    # Get configs
    model_config = get_model_config(args)
    input_meta_data = get_input_meta_data(data_dir)
    label_mapping = get_label_mapping(data_dir)
    logger.info(f'Loaded training data meta.json file: {input_meta_data}')

    # Calculate steps, warmup steps and eval steps
    train_data_size = input_meta_data['train_data_size']
    num_labels = input_meta_data['num_labels']
    max_seq_length = input_meta_data['max_seq_length']
    if args.limit_train_steps is None:
        steps_per_epoch = int(train_data_size / args.train_batch_size)
    else:
        steps_per_epoch = args.limit_train_steps
    warmup_proportion = 0.1
    warmup_steps = int(args.num_epochs * train_data_size * warmup_proportion/ args.train_batch_size)
    if args.limit_eval_steps is None:
        eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / args.eval_batch_size))
    else:
        eval_steps = args.limit_eval_steps
    logger.info(f'Running {args.num_epochs} epochs with {steps_per_epoch:,} steps per epoch')
    logger.info(f'Using warmup proportion of {warmup_proportion}, resulting in {warmup_steps:,} warmup steps')

    # Get model
    classifier_model, core_model = get_model(args, model_config, steps_per_epoch, warmup_steps, num_labels, max_seq_length)
    optimizer = classifier_model.optimizer
    loss_fn = get_loss_fn(num_labels)

    # Restore checkpoint
    if args.init_checkpoint is None:
        model_key = f'{args.model_class}_{args.model_type}'
        checkpoint_path = os.path.join(PRETRAINED_MODELS[model_key], 'bert_model.ckpt')
    else:
        checkpoint_path = f'gs://{args.bucket_name}/{args.project_name}/pretrain/runs/{args.init_checkpoint}'
    checkpoint = tf.train.Checkpoint(model=core_model)
    checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
    logger.info(f'Successfully restored checkpoint from {checkpoint_path}')

    # Run keras compile
    logger.info(f'Compiling keras model...')
    classifier_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=get_metrics(),
        experimental_steps_per_execution=args.steps_per_loop)
    logger.info(f'... done')

    # Create all custom callbacks
    summary_dir = os.path.join(output_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint_path = os.path.join(output_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
    time_history_callback = keras_utils.TimeHistory(
        batch_size=args.train_batch_size,
        log_steps=args.log_steps,
        logdir=output_dir)
    custom_callbacks = [summary_callback, checkpoint_callback, time_history_callback]
    if args.early_stopping_epochs > 0:
        logger.info(f'Using early stopping of after {args.early_stopping_epochs} epochs of val_loss not decreasing')
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=args.early_stopping_epochs, monitor='val_loss')
        custom_callbacks.append(early_stopping_callback)

    # Generate dataset_fn
    train_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecord', 'train.tfrecord'),
        max_seq_length,
        args.train_batch_size,
        is_training=True)
    eval_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecord', 'dev.tfrecord'),
        max_seq_length,
        args.eval_batch_size,
        is_training=False)

    # Add mertrics callback to calculate performance metrics at the end of epoch
    performance_metrics_callback = Metrics(eval_input_fn, label_mapping)
    custom_callbacks.append(performance_metrics_callback)

    # Run keras fit
    time_start = time.time()
    logger.info('Run training...')
    history = classifier_model.fit(
        x=train_input_fn(),
        validation_data=eval_input_fn(),
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)
    time_end = time.time()
    training_time_min = (time_end-time_start)/60
    logger.info(f'Finished training after {training_time_min:.1f} min')

    # Write log to Training Log File
    final_scores = performance_metrics_callback.scores[-1]
    all_scores = performance_metrics_callback.scores
    all_predictions = performance_metrics_callback.predictions
    data = {
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_name': run_name,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'max_seq_length': max_seq_length,
            'training_time_min': training_time_min,
            'data_dir': data_dir,
            'output_dir': output_dir,
            **history.history,
            **final_scores,
            **vars(args),
            'all_scores': all_scores,
            'all_predictions': all_predictions
            }
    f_path_log_remote = f'gs://{args.bucket_name}/{args.project_name}/finetune/traininglog.csv'
    f_path_log_local = 'traininglog.csv'
    append_to_csv(data, f_path_log_local, f_path_log_remote)

def main(args):
    # Get distribution strategy
    if args.not_use_tpu:
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored', num_gpus=args.num_gpus)
    else:
        logger.info(f'Intializing TPU on address {args.tpu_ip}...')
        tpu_address = f'grpc://{args.tpu_ip}:8470'
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='tpu', tpu_address=tpu_address, num_gpus=args.num_gpus)
    # Run training
    for repeat in range(args.repeats):
        train(args, strategy, repeat)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-projects', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--finetune_data', default='maternal_vaccine_stance_lshtm', choices=['maternal_vaccine_stance_lshtm',\
            'covid_worry', 'vaccine_sentiment_epfl', 'twittter_sentiment_semeval'],
            help='Finetune data folder name. The folder has to be located in gs://{bucket_name}/{project_name}/finetune/finetune_data/{finetune_data}.\
                    TFrecord files (train.tfrecord and dev.tfrecord as well as meta.json) should be located in a \
                    subfolder gs://{bucket_name}/{project_name}/finetune/finetune_data/{finetune_data}/tfrecord/')
    parser.add_argument('--tpu_ip', default='10.217.209.114', help='IP-address of the TPU')
    parser.add_argument('--not_use_tpu', action='store_true', default=False, help='Do not use TPUs')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--init_checkpoint', default=None, help='Run name to initialize checkpoint from. Example: "run2/ctl_step_8000.ckpt-8". \
            By default using a pretrained model from gs://cloud-tpu-checkpoints.')
    parser.add_argument('--init_checkpoint_index', type=int, help='Checkpoint index. This argument is ignored and only added for reporting.')
    parser.add_argument('--repeats', default=1, type=int, help='Number of times the script should run. Default is 1')
    parser.add_argument('--num_epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--limit_train_steps', type=int, help='Limit the number of train steps per epoch. Useful for testing.')
    parser.add_argument('--limit_eval_steps', type=int, help='Limit the number of eval steps per epoch. Useful for testing.')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--early_stopping_epochs', default=-1, type=int, help='Stop when loss hasn\'t decreased during n epochs')
    parser.add_argument('--model_class', default='bert', type=str, choices=['bert'], help='Model class')
    parser.add_argument('--model_type', default='large_uncased', choices=['large_uncased', 'base_uncased'], type=str, help='Model class')
    parser.add_argument('--optimizer_type', default='adamw', choices=['adamw', 'lamb'], type=str, help='Optimizer')
    parser.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16', 'fp16'], type=str, help='Data type')
    parser.add_argument('--steps_per_loop', default=100, type=int, help='Steps per loop')
    parser.add_argument('--log_steps', default=100, type=int, help='Frequency with which to log timing information with TimeHistory.')
    parser.add_argument('--model_config_path', default=None, type=str, help='Path to model config file, by default \
            try to infer from model_class/model_type args and fetch from gs://cloud-tpu-checkpoints')
    # Currently not supported:
    # parser.add_argument('--store_last_layer', action='store_true', default=False, help='Store last layer of encoder')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
