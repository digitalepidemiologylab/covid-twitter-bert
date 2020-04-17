#!/bin/sh

output_file="output/pretrain/$(basename $1).tfrecords.gz"

# PYTHONPATH="$(pwd)/../tensorflow_models" python ./tensorflow_models/official/nlp/data/create_pretraining_data.py --input_file "$1" --gzip_compress --max_seq_length 96 --output_file $output_file --vocab_file ./vocabs/bert-large-uncased-vocab.txt --max_predictions_per_seq 14 --helpfull
PYTHONPATH="$(pwd)/../tensorflow_models" python ../tensorflow_models/official/nlp/data/create_pretraining_data.py --helpfull


# --helpfull output
# ./tensorflow_models/official/nlp/data/create_pretraining_data.py:
#   --[no]do_lower_case: Whether to lower case the input text. Should be True for uncased models and False for cased models.
#     (default: 'true')
#   --[no]do_whole_word_mask: Whether to use whole word masking rather than per-WordPiece masking.
#     (default: 'false')
#   --dupe_factor: Number of times to duplicate the input data (with different masks).
#     (default: '10')
#     (an integer)
#   --[no]gzip_compress: Whether to use `GZIP` compress option to get compressed TFRecord files.
#     (default: 'false')
#   --input_file: Input raw text file (or comma-separated list of files).
#   --masked_lm_prob: Masked LM probability.
#     (default: '0.15')
#     (a number)
#   --max_predictions_per_seq: Maximum number of masked LM predictions per sequence.
#     (default: '20')
#     (an integer)
#   --max_seq_length: Maximum sequence length.
#     (default: '128')
#     (an integer)
#   --output_file: Output TF example file (or comma-separated list of files).
#   --random_seed: Random seed for data generation.
#     (default: '12345')
#     (an integer)
#   --short_seq_prob: Probability of creating sequences which are shorter than the maximum length.
#     (default: '0.1')
#     (a number)
#   --vocab_file: The vocabulary file that the BERT model was trained on.
#
# absl.app:
#   -?,--[no]help: show this help
#     (default: 'false')
#   --[no]helpfull: show full help
#     (default: 'false')
#   --[no]helpshort: show this help
#     (default: 'false')
#   --[no]helpxml: like --helpfull, but generates XML output
#     (default: 'false')
#   --[no]only_check_args: Set to true to validate args and exit.
#     (default: 'false')
#   --[no]pdb_post_mortem: Set to true to handle uncaught exceptions with PDB post mortem.
#     (default: 'false')
#   --profile_file: Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
#   --[no]run_with_pdb: Set to true for PDB debug mode
#     (default: 'false')
#   --[no]run_with_profiling: Set to true for profiling the script. Execution will be slower, and the output format might change over time.
#     (default: 'false')
#   --[no]use_cprofile_for_profiling: Use cProfile instead of the profile module for profiling. This has no effect unless --run_with_profiling is set.
#     (default: 'true')
#
# absl.logging:
#   --[no]alsologtostderr: also log to stderr?
#     (default: 'false')
#   --log_dir: directory to write logfiles into
#     (default: '')
#   --[no]logtostderr: Should only log to stderr?
#     (default: 'false')
#   --[no]showprefixforinfo: If False, do not prepend prefix to info messages when it's logged to stderr, --verbosity is set to INFO level, and python logging is used.
#     (default: 'true')
#   --stderrthreshold: log messages at this level, or more severe, to stderr in addition to the logfile.  Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes --alsologtostderr. Using --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to --verbosity and requires logfile not be stderr.
#     (default: 'fatal')
#   -v,--verbosity: Logging verbosity level. Messages logged at this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied, the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed.
#     (default: '-1')
#     (an integer)
#
# absl.testing.absltest:
#   --test_random_seed: Random seed for testing. Some test frameworks may change the default value of this flag between runs, so it is not appropriate for seeding probabilistic tests.
#     (default: '301')
#     (an integer)
#   --test_randomize_ordering_seed: If positive, use this as a seed to randomize the execution order for test cases. If "random", pick a random seed to use. If 0 or not set, do not randomize test case execution order. This flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
#     (default: '')
#   --test_srcdir: Root of directory tree where source files live
#     (default: '')
#   --test_tmpdir: Directory for temporary testing files
#     (default: '/tmp/absl_testing')
#   --xml_output_file: File to store XML test results
#     (default: '')
#
# tensorflow.python.ops.parallel_for.pfor:
#   --[no]op_conversion_fallback_to_while_loop: If true, falls back to using a while loop for ops for which a converter is not defined.
#     (default: 'false')
#
# absl.flags:
#   --flagfile: Insert flag definitions from the given file into the command line.
#     (default: '')
#   --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that have arguments MUST use the --flag=value format.
#     (default: '')
