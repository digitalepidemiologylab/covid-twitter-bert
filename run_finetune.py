import sys
# import from official repo
sys.path.append('tensorflow_models')
from official.nlp.bert import bert_models
from official.utils.misc import distribution_utils
from official.nlp.bert import configs as bert_configs
from official.modeling import performance
from official.nlp.bert import input_pipeline
from official.utils.misc import keras_utils

import os
import datetime
import time
import argparse
import math
import logging
from logging.handlers import RotatingFileHandler
import tqdm
import json
import tensorflow as tf
from utils.misc import ArgParseDefault, save_to_json, add_bool_arg
from utils.finetune_helpers import Metrics
import utils.optimizer
from config import PRETRAINED_MODELS


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# remove duplicate logger (not sure why this is happening, possibly an issue with the imports)
tf_logger = tf.get_logger()
tf_logger.handlers.pop()

# add file logging
handler = RotatingFileHandler("logs/finetune.log", maxBytes=2000, backupCount=10)
logger.addHandler(handler)

def get_model_config(config_path):
    config = bert_configs.BertConfig.from_json_file(config_path)
    return config

def get_input_meta_data(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, 'meta.json'), 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    return input_meta_data

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

def get_model(args, model_config, steps_per_epoch, warmup_steps, num_labels, max_seq_length, is_hub_module=False):
    # Get classifier and core model (used to initialize from checkpoint)
    if args.init_checkpoint is None and PRETRAINED_MODELS[args.model_class]['is_tfhub_model']:
        # load pretrained model from TF-hub
        hub_module_url = f"https://tfhub.dev/{PRETRAINED_MODELS[args.model_class]['hub_url']}"
        hub_module_trainable = True
    else:
        hub_module_url = None
        hub_module_trainable = False
    classifier_model, core_model = bert_models.classifier_model(
            model_config,
            num_labels,
            max_seq_length,
            hub_module_url=hub_module_url,
            hub_module_trainable=hub_module_trainable)
    # Optimizer
    optimizer = utils.optimizer.create_optimizer(
            args.learning_rate,
            steps_per_epoch * args.num_epochs,
            warmup_steps,
            args.end_lr,
            args.optimizer_type)
    classifier_model.optimizer = configure_optimizer(
            optimizer,
            use_float16=False,
            use_graph_rewrite=False)
    return classifier_model, core_model

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training=True):
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
    return [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]

def get_model_config_path(args):
    try:
        config_path = PRETRAINED_MODELS[args.model_class]['config']
    except KeyError:
        raise ValueError(f'Could not find a pretrained model matching the model class {args.model_class}')
    return os.path.join('configs', config_path)

def get_run_name(args):
    # Use timestamp to generate a unique run name
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    if args.run_prefix:
        run_name = f'run_{ts}_{args.run_prefix}'
    else:
        run_name = f'run_{ts}'
    return run_name

def run(args):
    """Train using the Keras/TF 2.0. Adapted from the tensorflow/models Github"""
    # CONFIG
    run_name = get_run_name(args)
    logger.info(f'*** Starting run {run_name} ***')
    data_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/finetune_data/{args.finetune_data}'
    output_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/runs/{run_name}'

    # Get configs
    pretrained_model_config_path = get_model_config_path(args)
    model_config = get_model_config(pretrained_model_config_path)

    # Meta data/label mapping
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
    warmup_steps = int(args.num_epochs * train_data_size * args.warmup_proportion/ args.train_batch_size)
    if args.limit_eval_steps is None:
        eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / args.eval_batch_size))
    else:
        eval_steps = args.limit_eval_steps

    # some logging
    if args.init_checkpoint is None:
        logger.info(f'Finetuning on datset {args.finetune_data} using default pretrained model {args.model_class}')
    else:
        logger.info(f'Finetuning on datset {args.finetune_data} using pretrained model in {args.init_checkpoint} of type {args.model_class}')
    logger.info(f'Running {args.num_epochs} epochs with {steps_per_epoch:,} steps per epoch')
    logger.info(f'Using warmup proportion of {args.warmup_proportion}, resulting in {warmup_steps:,} warmup steps')
    logger.info(f'Using learning rate: {args.learning_rate}, training batch size: {args.train_batch_size}, num_epochs: {args.num_epochs}')

    # Get model
    classifier_model, core_model = get_model(args, model_config, steps_per_epoch, warmup_steps, num_labels, max_seq_length)
    optimizer = classifier_model.optimizer
    loss_fn = get_loss_fn(num_labels)

    # Restore checkpoint
    if args.init_checkpoint:
        checkpoint_path = f'gs://{args.bucket_name}/{args.project_name}/pretrain/runs/{args.init_checkpoint}'
        checkpoint = tf.train.Checkpoint(model=core_model)
        checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
        logger.info(f'Successfully restored checkpoint from {checkpoint_path}')

    # Run keras compile
    logger.info(f'Compiling keras model...')
    classifier_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=get_metrics())
    logger.info(f'... done')

    # Create all custom callbacks
    summary_dir = os.path.join(output_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir, profile_batch=0)
    checkpoint_path = os.path.join(output_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
    time_history_callback = keras_utils.TimeHistory(
        batch_size=args.train_batch_size,
        log_steps=args.time_history_log_steps,
        logdir=summary_dir)
    custom_callbacks = [summary_callback, time_history_callback]
    if args.save_model:
        custom_callbacks.append(checkpoint_callback)
    if args.early_stopping_epochs > 0:
        logger.info(f'Using early stopping of after {args.early_stopping_epochs} epochs of val_loss not decreasing')
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=args.early_stopping_epochs, monitor='val_loss')
        custom_callbacks.append(early_stopping_callback)

    # Generate dataset_fn
    train_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecords', 'train.tfrecords'),
        max_seq_length,
        args.train_batch_size,
        is_training=True)
    eval_input_fn = get_dataset_fn(
        os.path.join(data_dir, 'tfrecords', 'dev.tfrecords'),
        max_seq_length,
        args.eval_batch_size,
        is_training=False)

    # Add mertrics callback to calculate performance metrics at the end of epoch
    performance_metrics_callback = Metrics(
            eval_input_fn,
            label_mapping,
            os.path.join(summary_dir, 'metrics'),
            eval_steps,
            args.eval_batch_size,
            args.validation_freq)
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
        validation_freq=args.validation_freq,
        callbacks=custom_callbacks,
        verbose=1)
    time_end = time.time()
    training_time_min = (time_end-time_start)/60
    logger.info(f'Finished training after {training_time_min:.1f} min')

    # Write training log
    all_scores = performance_metrics_callback.scores
    all_predictions = performance_metrics_callback.predictions
    if len(all_scores) > 0:
        final_scores = all_scores[-1]
        logger.info(f'Final eval scores: {final_scores}')
    else:
        final_scores = {}
    full_history = history.history
    if len(full_history) > 0:
        final_val_loss = full_history['val_loss'][-1]
        final_loss = full_history['loss'][-1]
        logger.info(f'Final training loss: {final_loss:.2f}, Final validation loss: {final_val_loss:.2f}')
    else:
        final_val_loss = None
        final_loss = None
    data = {
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_name': run_name,
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'max_seq_length': max_seq_length,
            'num_train_steps': steps_per_epoch * args.num_epochs,
            'eval_steps': eval_steps,
            'steps_per_epoch': steps_per_epoch,
            'training_time_min': training_time_min,
            'data_dir': data_dir,
            'output_dir': output_dir,
            'all_scores': all_scores,
            'all_predictions': all_predictions,
            'num_labels': num_labels,
            'label_mapping': label_mapping,
            **full_history,
            **final_scores,
            **vars(args),
            }
    # Write run_log
    f_path_training_log = os.path.join(output_dir, 'run_logs.json')
    logger.info(f'Writing training log to {f_path_training_log}...')
    save_to_json(data, f_path_training_log)
    # Write bert config
    model_config.id2label = label_mapping
    model_config.label2id = {v:k for k, v in label_mapping.items()}
    model_config.max_seq_length = max_seq_length
    model_config.num_labels = num_labels
    f_path_bert_config = os.path.join(output_dir, 'bert_config.json')
    logger.info(f'Writing BERT config to {f_path_bert_config}...')
    save_to_json(model_config.to_dict(), f_path_bert_config)

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

def main(args):
    # Set TF Hub caching to bucket
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(f'gs://{args.bucket_name}/tmp')
    # Get distribution strategy
    if args.use_tpu:
        if args.tpu_ip:
            logger.info(f'Intializing TPU on address {args.tpu_ip}...')
            tpu_address = f'grpc://{args.tpu_ip}:8470'
            strategy = distribution_utils.get_distribution_strategy(distribution_strategy='tpu', tpu_address=tpu_address)
        elif args.tpu_name:
            logger.info(f'Intializing TPU with name {args.tpu_name}...')
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name)
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        else:
            raise ValueError(f'You need to either specify a tpu_ip or a tpu_name in order to use a TPU.')
    else:
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored', num_gpus=args.num_gpus)
    # set mixed precision
    set_mixed_precision_policy(args)
    # Run training
    for repeat in range(args.repeats):
        with strategy.scope():
            run(args)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--finetune_data', required=True, help='Finetune data folder sub path. Path has to be in gs://{bucket_name}/{project_name}/finetune/finetune_data/{finetune_data}.\
                    This folder includes a meta.json (containing meta info about the dataset), and a file label_mapping.json. \
                    TFrecord files (train.tfrecords and dev.tfrecords) should be located in a \
                    subfolder gs://{bucket_name}/{project_name}/finetune/finetune_data/{finetune_data}/tfrecords/')
    parser.add_argument('--bucket_name', required=True, help='Bucket name')
    parser.add_argument('--tpu_ip', required=False, help='IP-address of the TPU')
    parser.add_argument('--tpu_name', required=False, help='Name of the TPU (required for pods)')
    parser.add_argument('--run_prefix', help='Prefix to be added to all runs. Useful to group runs')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--model_class', default='bert_large_uncased_wwm', choices=PRETRAINED_MODELS.keys(), help='Model class to use')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--init_checkpoint', default=None, help='Run name to initialize checkpoint from. Example: "run2/ctl_step_8000.ckpt-8". \
            By default using a pretrained model from gs://{bucket_name}/pretrained_models/')
    parser.add_argument('--init_checkpoint_index', type=int, help='Checkpoint index. This argument is ignored and only added for reporting.')
    parser.add_argument('--repeats', default=1, type=int, help='Number of times the script should run. Default is 1')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs')
    parser.add_argument('--limit_train_steps', type=int, help='Limit the number of train steps per epoch. Useful for testing.')
    parser.add_argument('--limit_eval_steps', type=int, help='Limit the number of eval steps per epoch. Useful for testing.')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--end_lr', default=0, type=float, help='Final learning rate')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Learning rate warmup proportion')
    parser.add_argument('--max_seq_length', default=96, type=int, help='Maximum sequence length')
    parser.add_argument('--early_stopping_epochs', default=-1, type=int, help='Stop when loss hasn\'t decreased during n epochs')
    parser.add_argument('--optimizer_type', default='adamw', choices=['adamw', 'lamb'], type=str, help='Optimizer')
    parser.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16', 'fp16'], type=str, help='Data type')
    parser.add_argument('--steps_per_loop', default=10, type=int, help='Steps per loop (unavailable for Keras fit in TF 2.2, will be added in later version)')
    parser.add_argument('--validation_freq', default=None, type=int, nargs='+', help='Validation frequency. Run eval after specified epochs. Default: After every epoch')
    parser.add_argument('--time_history_log_steps', default=10, type=int, help='Frequency with which to log timing information with TimeHistory.')
    add_bool_arg(parser, 'use_tpu', default=True, help='Use TPU')
    add_bool_arg(parser, 'save_model', default=True, help='Save model checkpoint(s)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
