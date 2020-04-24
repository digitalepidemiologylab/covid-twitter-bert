import sys
# import from official repo
sys.path.append('tensorflow_models')

from official.nlp.bert.tf1_checkpoint_converter_lib import convert, BERT_V2_NAME_REPLACEMENTS, BERT_NAME_REPLACEMENTS, BERT_PERMUTATIONS, BERT_V2_PERMUTATIONS
from utils.misc import ArgParseDefault


def main(args):
    convert(args.input_checkpoint, args.output_checkpoint, args.num_heads, args.name_replacements, args.name_permutations)

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--input_checkpoint', required=True, help='Path to v1 checkpoint')
    parser.add_argument('--output_checkpoint', required=True, help='Path to checkpoint to be written out.')
    parser.add_argument('--num_heads', default=16, help='Path to checkpoint to be written out.')
    parser.add_argument('--name_replacements', default=BERT_NAME_REPLACEMENTS, help='Name replacements')
    parser.add_argument('--name_permutations', default=BERT_PERMUTATIONS, help='Name permuations')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
