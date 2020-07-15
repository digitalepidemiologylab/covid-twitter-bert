"""
This script can be used to test the converted models
"""
import argparse
from transformers import BertModel, BertConfig
import torch
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

def validate_model(bert_original, bert_converted):
    assert bert_original.num_parameters() == bert_converted.num_parameters()
    logger.info(f'Both models have {bert_original.num_parameters():,} parameters')

    assert len(bert_original.state_dict()) == len(bert_converted.state_dict())
    logger.info(f'Both models have {len(bert_original.state_dict()):,} layers')

    for (layer_original, value_original), (layer_converted, value_converted) in zip(bert_original.state_dict().items(), bert_converted.state_dict().items()):
        assert layer_original == layer_converted
        if not torch.eq(value_original, value_converted).all():
            raise ValueError(f'Incorrect weights for {layer_original}')
    logger.info('Success! Both models are identical!')

def validate(args):
    if args.model_type == 'BertModel':
        model_cls = BertModel
    elif args.model_type == 'BertForPreTraining':
        model_cls = BertForPreTraining

    logger.info(f'Loading converted model {args.pytorch_model_path}')
    bert_converted = model_cls.from_pretrained(f'./{args.pytorch_model_path}')

    logger.info(f'Loading original model {args.validate_against}')
    bert_original = model_cls.from_pretrained(args.validate_against)

    logger.info('Validating model...')
    validate_model(bert_original, bert_converted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_model_path", type=str, required=True, help="Path to PyTorch model to validate")
    parser.add_argument("--validate_against", choices=['bert-large-uncased', 'bert-base-uncased', 'bert-large-uncased-whole-word-masking'], required=True,
            help="Official pretrained model to validate against. Model in pytorch_model_path will be compared against this model type.")
    parser.add_argument("--model_type", choices=['BertModel', 'BertForPreTraining'], default='BertModel', help="Model architecture to validate")
    args = parser.parse_args()
    validate(args)

if __name__ == "__main__":
    main()
