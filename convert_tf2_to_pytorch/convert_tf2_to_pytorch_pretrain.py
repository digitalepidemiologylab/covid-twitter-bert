"""
This script converts a pretrained TF2.x Bert model (including a MLM and NSP head) to Pytorch.

Note: The most recent pretrained model architectures on the official GitHub (https://github.com/tensorflow/models/tree/master/official/nlp/bert)
have different variable names. This script only works with models trained with code from the following commit state:
    https://github.com/tensorflow/models/tree/93490036e00f37ecbe6693b9ff4ae488bb8e9270
such as COVID-Twitter-BERT. More recent commits have slightly different variable names for the MLM/NSP layers and the embeddings.
"""

import argparse
import re
import os
import logging
import json
import torch
import tensorflow as tf
from transformers import BertConfig, BertForPreTraining


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    layer_depth = []
    for full_name, shape in init_vars:
        # logger.info("Loading TF weight {} with shape {}".format(name, shape))
        name = full_name.split("/")
        if full_name == '_CHECKPOINTABLE_OBJECT_GRAPH' or name[0] in ['global_step', 'save_counter']:
            logger.info(f'Skipping non-model layer {full_name}')
            continue
        if 'optimizer' in full_name:
            logger.info(f'Skipping optimization layer {full_name}')
            continue
        if name[0] == 'model':
            # ignore initial 'model'
            name = name[1:]
        # figure out how many levels deep the name is
        depth = 0
        for _name in name:
            if _name.startswith('layer_with_weights'):
                depth += 1
            else:
                break
        layer_depth.append(depth)
        # read data
        array = tf.train.load_variable(tf_path, full_name)
        names.append('/'.join(name))
        arrays.append(array)
    logger.info(f'Read a total of {len(arrays):,} layers')

    if len(set(layer_depth)) != 1:
        raise ValueError(f'Found layer names with different depths (layer depth {list(set(layer_depth))})')
    layer_depth = list(set(layer_depth))[0]
    if layer_depth != 3:
        raise ValueError(f'The checkpoint file does not contain the correct layer depth for Pretrained Bert models.')

    # convert layers
    logger.info('Converting weights...')
    for full_name, array in zip(names, arrays):
        name = full_name.split("/")
        # layer flags
        is_mlm_layer = False
        is_nsp_layer = False
        # find corresponding model layer
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            if m_name == '.ATTRIBUTES':
                # variable names end with .ATTRIBUTES/VARIABLE_VALUE
                break
            if m_name.startswith('layer_with_weights'):
                layer_num = int(m_name.split('-')[-1])
                if i == 0:
                    # ignore first occurrence of layer_with_weights
                    continue
                elif i == 1:
                    # corresponds to encoder/MLM head/NSP head
                    if layer_num == 0:
                        # transformer/encoder
                        trace.append('bert')
                        pointer = getattr(pointer, 'bert')
                    elif layer_num == 1:
                        # MLM head
                        is_mlm_layer = True
                        trace.extend(['cls', 'predictions'])
                        pointer = getattr(pointer, 'cls')
                        pointer = getattr(pointer, 'predictions')
                    elif layer_num == 2:
                        # NSP head
                        is_nsp_layer = True
                        trace.extend(['cls', 'seq_relationship'])
                        pointer = getattr(pointer, 'cls')
                        pointer = getattr(pointer, 'seq_relationship')
                    else:
                        raise ValueError(f'Unknown level {i} in layer {full_name}')
                elif i == 2:
                    if is_mlm_layer:
                        if layer_num == 0:
                            trace.extend(['transform', 'dense'])
                            pointer = getattr(pointer, 'transform')
                            pointer = getattr(pointer, 'dense')
                        elif layer_num == 1:
                            trace.extend(['transform', 'LayerNorm'])
                            pointer = getattr(pointer, 'transform')
                            pointer = getattr(pointer, 'LayerNorm')
                        elif layer_num == 2:
                            trace.append('decoder')
                            pointer = getattr(pointer, 'decoder')
                        else:
                            raise ValueError(f'Unexpected layer {full_name}')
                    elif is_nsp_layer:
                        # there is only a single layer for NSP
                        continue
                    else:
                        if layer_num <= 2:
                            # embedding layers
                            # layer_num 0: word_embeddings
                            # layer_num 1: position_embeddings
                            # layer_num 2: token_type_embeddings
                            continue
                        elif layer_num == 3:
                            # embedding LayerNorm
                            trace.extend(['embeddings', 'LayerNorm'])
                            pointer = getattr(pointer, 'embeddings')
                            pointer = getattr(pointer, 'LayerNorm')
                        elif layer_num > 3  and layer_num < config.num_hidden_layers + 4:
                            # encoder layers
                            trace.extend(['encoder', 'layer', str(layer_num - 4)])
                            pointer = getattr(pointer, 'encoder')
                            pointer = getattr(pointer, 'layer')
                            pointer = pointer[layer_num - 4]
                        elif layer_num == config.num_hidden_layers + 4:
                            # pooler layer
                            trace.extend(['pooler', 'dense'])
                            pointer = getattr(pointer, 'pooler')
                            pointer = getattr(pointer, 'dense')

            elif m_name == 'embeddings':
                trace.append('embeddings')
                pointer = getattr(pointer, 'embeddings')
                if layer_num == 0:
                    trace.append('word_embeddings')
                    pointer = getattr(pointer, 'word_embeddings')
                elif layer_num == 1:
                    trace.append('position_embeddings')
                    pointer = getattr(pointer, 'position_embeddings')
                elif layer_num == 2:
                    trace.append('token_type_embeddings')
                    pointer = getattr(pointer, 'token_type_embeddings')
                else:
                    raise ValueError('Unknown embedding layer with name {full_name}')
                trace.append('weight')
                pointer = getattr(pointer, 'weight')
            elif m_name == '_attention_layer':
                # self-attention layer
                trace.extend(['attention', 'self'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'self')
            elif m_name == '_attention_layer_norm':
                # output attention norm
                trace.extend(['attention', 'output', 'LayerNorm'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'LayerNorm')
            elif m_name == '_attention_output_dense':
                # output attention dense
                trace.extend(['attention', 'output', 'dense'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_dense':
                # output dense
                trace.extend(['output', 'dense'])
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_layer_norm':
                # output dense
                trace.extend(['output', 'LayerNorm'])
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'LayerNorm')
            elif m_name == '_key_dense':
                # attention key
                trace.append('key')
                pointer = getattr(pointer, 'key')
            elif m_name == '_query_dense':
                # attention query
                trace.append('query')
                pointer = getattr(pointer, 'query')
            elif m_name == '_value_dense':
                # attention value
                trace.append('value')
                pointer = getattr(pointer, 'value')
            elif m_name == '_intermediate_dense':
                # attention intermediate dense
                trace.extend(['intermediate', 'dense'])
                pointer = getattr(pointer, 'intermediate')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_layer_norm':
                # output layer norm
                trace.append('output')
                pointer = getattr(pointer, 'output')
            # weights & biases
            elif m_name in ['bias', 'beta']:
                trace.append('bias')
                pointer = getattr(pointer, 'bias')
            elif m_name in ['kernel', 'gamma']:
                trace.append('weight')
                pointer = getattr(pointer, 'weight')
            else:
                raise ValueError(f'Unknown name {m_name}')
        trace = '.'.join(trace)
        # for certain layers reshape is necessary
        if (re.match(r'(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)', trace) or
                re.match(r'(\S+)\.attention\.output\.dense\.weight', trace)
                ):
            array = array.reshape(pointer.data.shape)
        if 'kernel' in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = torch.from_numpy(array)
        else:
            raise ValueError(f'Shape mismatch in layer {full_name}: Model expects shape {pointer.shape} but layer contains shape {array.shape}:')
        logger.info(f'Successfully set variable {full_name} to PyTorch layer {trace}')
    return model


def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, output_folder):
    # Instantiate model
    logger.info(f'Loading model based on config from {config_path}...')
    config = BertConfig.from_json_file(config_path)
    model = BertForPreTraining(config)

    # Load weights from checkpoint
    logger.info(f'Loading weights from checkpoint {tf_checkpoint_path}...')
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # Create dirs
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Save pytorch-model
    f_out_model = os.path.join(output_folder, 'pytorch_model.bin')
    logger.info(f'Saving PyTorch model to {f_out_model}...')
    torch.save(model.state_dict(), f_out_model)

    # Save config to output
    f_out_config = os.path.join(output_folder, 'config.json')
    logger.info(f'Saving config to {f_out_config}...')
    config.to_json_file(f_out_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to folder where PyTorch model will be stored (created if non-existant)."
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.output_folder)
