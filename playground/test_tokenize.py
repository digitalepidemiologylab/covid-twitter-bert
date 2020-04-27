import sys
sys.path.append('../tensorflow_models')
from official.nlp.bert import tokenization
import argparse
import os

vocab_file = os.path.join('vocabs', 'bert-large-uncased-vocab.txt')

def main(args):
    tknzr = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    print('Input:')
    print(args.input)
    print('\nTokenized:')
    tokenized = tknzr.tokenize(args.input)
    print(tokenized)
    print('\nTokenized and converted to IDs:')
    ids = tknzr.convert_tokens_to_ids(tokenized)
    print(ids)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help="Input to tokenize")
    parser.add_argument('--vocab_file', default=vocab_file, help="Path to vocab file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
