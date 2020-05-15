#!/bin/sh

BUCKET_NAME=MY-BUCKET-NAME
PROJECT_BUCKET=gs://${BUCKET_NAME}/covid-bert
TPU_ADDRESS=10.74.219.210
RUN_NAME=run2
PRETRAINED_MODEL=gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16

gsutil rm -r $PROJECT_BUCKET/pretrain/runs/${RUN_NAME}

PYTHONPATH="$(pwd)/../tensorflow_models" python ../tensorflow_models/official/nlp/bert/run_pretraining.py \
  --input_files ${PROJECT_BUCKET}/pretrain/pretrain_data/v1/pretrain_anonymized_bert_train_???.txt.tfrecords	\
  --max_seq_length 96 \
  --max_predictions_per_seq 14 \
  --num_train_epochs 10 \
  --num_steps_per_epoch 10000 \
  --train_batch_size 1024 \
  --tpu grpc://${TPU_ADDRESS}:8470 \
  --distribution_strategy tpu \
  --model_export_path ${PROJECT_BUCKET}/pretrain/runs/${RUN_NAME} \
  --model_dir ${PROJECT_BUCKET}/pretrain/runs/${RUN_NAME} \
  --init_checkpoint ${PRETRAINED_MODEL}/bert_model.ckpt \
  --bert_config_file ${PRETRAINED_MODEL}/bert_config.json \
  --steps_per_loop 1000 \
  --verbosity 0

# ../tensorflow_models/official/nlp/bert/run_pretraining.py:
#   --input_files: File path to retrieve training data for pre-training.
#   --max_predictions_per_seq: Maximum predictions per sequence_output.
#     (default: '20')
#     (an integer)
#   --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
#     (default: '128')
#     (an integer)
#   --num_steps_per_epoch: Total number of training steps to run per epoch.
#     (default: '1000')
#     (an integer)
#   --train_batch_size: Total batch size for training.
#     (default: '32')
#     (an integer)
#   --[no]use_next_sentence_label: Whether to use next sentence label to compute final loss.
#     (default: 'true')
#   --warmup_steps: Warmup steps for Adam weight decay optimizer.
#     (default: '10000.0')
#     (a number)
#
# absl.app:
#   -?,--[no]help: show this help
#     (default: 'false')
#   --[no]helpfull: show full help
#     (default: 'false')
#   -h,--[no]helpshort: show this help
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
# official.nlp.bert.common_flags:
#   --bert_config_file: Bert configuration file to define core bert layers.
#   --end_lr: The end learning rate for learning rate decay.
#     (default: '0.0')
#     (a number)
#   --gin_file: List of paths to the config files.;
#     repeat this option to specify a list of values
#   --gin_param: Newline separated list of Gin parameter bindings.;
#     repeat this option to specify a list of values
#   --[no]hub_module_trainable: True to make keras layers in the hub module trainable.
#     (default: 'true')
#   --hub_module_url: TF-Hub path/url to Bert module. If specified, init_checkpoint flag should not be used.
#   --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
#   --learning_rate: The initial learning rate for Adam.
#     (default: '5e-05')
#     (a number)
#   --model_export_path: Path to the directory, where trainined model will be exported.
#   --num_train_epochs: Total number of training epochs to perform.
#     (default: '3')
#     (an integer)
#   --optimizer_type: The type of optimizer to use for training (adamw|lamb)
#     (default: 'adamw')
#   --[no]scale_loss: Whether to divide the loss by number of replica inside the per-replica loss function.
#     (default: 'false')
#   --steps_per_loop: Number of steps per graph-mode loop. Only training step happens inside the loop. Callbacks will not be called inside.
#     (default: '1')
#     (an integer)
#   --tpu: TPU address to connect to.
#     (default: '')
#   --[no]use_keras_compile_fit: If True, uses Keras compile/fit() API for training logic. Otherwise use custom training loop.
#     (default: 'false')
# official.utils.flags._base:
#   -ds,--distribution_strategy:
#     The Distribution Strategy to use for training. Accepted values are 'off',
#     'one_device', 'mirrored', 'parameter_server', 'collective', case insensitive.
#     'off' means not to use Distribution Strategy; 'default' means to choose from
#     `MirroredStrategy` or `OneDeviceStrategy` according to the number of GPUs.
#     (default: 'mirrored')
#   -md,--model_dir:
#     The location of the model checkpoint files.
#     (default: '/tmp')
#   -ng,--num_gpus:
#     How many GPUs to use at each worker with the DistributionStrategies API. The
#     default is 1.
#     (default: '1')
#     (an integer)
#   --[no]run_eagerly: Run the model op by op without building a model function.
#     (default: 'false')
#
# official.utils.flags._benchmark:
#   --log_steps: Frequency with which to log timing information with TimeHistory.
#     (default: '100')
#     (an integer)
#
# official.utils.flags._distribution:
#   --task_index:
#     If multi-worker training, the task_index of this worker.
#     (default: '-1')
#     (an integer)
#   --worker_hosts:
#     Comma-separated list of worker ip:port pairs for running multi-worker models
#     with DistributionStrategy.  The user would start the program on each host with
#     identical value for this flag.
#
# official.utils.flags._performance:
#   -ara,--all_reduce_alg:
#     Defines the algorithm to use for performing all-reduce.When specified with
#     MirroredStrategy for single worker, this controls
#     tf.contrib.distribute.AllReduceCrossTowerOps.  When specified with
#     MultiWorkerMirroredStrategy, this controls
#     tf.distribute.experimental.CollectiveCommunication; valid options are `ring` and
#     `nccl`.
#   --datasets_num_private_threads:
#     Number of threads for a private threadpool created for alldatasets
#     computation..
#     (an integer)
#   -dt,--dtype: <fp16|bf16|fp32>:
#     The TensorFlow datatype used for calculations. Variables may be cast to a
#     higher precision on a case-by-case basis for numerical stability.
#     (default: 'fp32')
#   --[no]enable_xla: Whether to enable XLA auto jit compilation
#     (default: 'false')
#   --fp16_implementation: <k|e|r|a|s|'|,| |'|g|r|a|p|h|_|r|e|w|r|i|t|e>:
#     When --dtype=fp16, how fp16 should be implemented. This has no impact on
#     correctness. 'keras' uses the tf.keras.mixed_precision API. 'graph_rewrite' uses
#     the tf.train.experimental.enable_mixed_precision_graph_rewrite API.
#     (default: 'keras')
#   -ls,--loss_scale:
#     The amount to scale the loss by when the model is run. This can be an int/float
#     or the string 'dynamic'. Before gradients are computed, the loss is multiplied
#     by the loss scale, making all gradients loss_scale times larger. To adjust for
#     this, gradients are divided by the loss scale before being applied to variables.
#     This is mathematically equivalent to training without a loss scale, but the loss
#     scale helps avoid some intermediate gradients from underflowing to zero. If not
#     provided the default for fp16 is 128 and 1 for all other dtypes. The string
#     'dynamic' can be used to dynamically determine the optimal loss scale during
#     training, but currently this significantly slows down performance
#   -pgtc,--per_gpu_thread_count:
#     The number of threads to use for GPU. Only valid when tf_gpu_thread_mode is not
#     global.
#     (default: '0')
#     (an integer)
#   -gt_mode,--tf_gpu_thread_mode:
#     Whether and how the GPU device uses its own threadpool.
#
# tensorflow.python.ops.parallel_for.pfor:
#   --[no]op_conversion_fallback_to_while_loop: DEPRECATED: Flag is ignored.
#     (default: 'true')
#
# tensorflow_hub.resolver:
#   --tfhub_cache_dir: If set, TF-Hub will download and cache Modules into this directory. Otherwise it will attempt to find a network path.
#
# absl.flags:
#   --flagfile: Insert flag definitions from the given file into the command line.
#     (default: '')
#   --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that have arguments MUST use the --flag=value format.
#     (default: '')

