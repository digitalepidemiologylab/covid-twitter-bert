#!/bin/sh

# Help of create pretrain data:
# --[no]do_lower_case: Whether to lower case the input text. Should be True for uncased models and False for cased models.
#   (default: 'true')
# --[no]do_whole_word_mask: Whether to use whole word masking rather than per-WordPiece masking.
#   (default: 'false')
# --dupe_factor: Number of times to duplicate the input data (with different masks).
#   (default: '10')
#   (an integer)
# --[no]gzip_compress: Whether to use `GZIP` compress option to get compressed TFRecord files.
#   (default: 'false')
# --input_file: Input raw text file (or comma-separated list of files).
# --masked_lm_prob: Masked LM probability.
#   (default: '0.15')
#   (a number)
# --max_predictions_per_seq: Maximum number of masked LM predictions per sequence.
#   (default: '20')
#   (an integer)
# --max_seq_length: Maximum sequence length.
#   (default: '128')
#   (an integer)
# --output_file: Output TF example file (or comma-separated list of files).
# --random_seed: Random seed for data generation.
#   (default: '12345')
#   (an integer)
# --short_seq_prob: Probability of creating sequences which are shorter than the maximum length.
#   (default: '0.1')
#   (a number)
# --vocab_file: The vocabulary file that the BERT model was trained on.

output_file="output/pretrain/$(basename $1).tfrecords.gz"

PYTHONPATH="$(pwd)/tensorflow_models" python ./tensorflow_models/official/nlp/data/create_pretraining_data.py --input_file "$1" --gzip_compress --max_seq_length 96 --output_file $output_file --vocab_file ./vocabs/bert-large-uncased-vocab.txt --max_predictions_per_seq 14
