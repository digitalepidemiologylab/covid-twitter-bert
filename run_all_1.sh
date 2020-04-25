#!/usr/bin/env sh

TPU_IP=10.74.219.210
NUM_EPOCHS=10
# INIT_CHECKPOINT_1=run2/pretrained/bert_model_step_10000.ckpt-1
# INIT_CHECKPOINT_2=run2/pretrained/bert_model_step_20000.ckpt-2
# INIT_CHECKPOINT_3=run2/pretrained/bert_model_step_30000.ckpt-3
# INIT_CHECKPOINT_5=run2/pretrained/bert_model_step_50000.ckpt-5
# INIT_CHECKPOINT_6=run2/pretrained/bert_model_step_60000.ckpt-6
# INIT_CHECKPOINT_7=run2/pretrained/bert_model_step_70000.ckpt-7
# INIT_CHECKPOINT_8=run2/pretrained/bert_model_step_80000.ckpt-8
# INIT_CHECKPOINT_9=run2/pretrained/bert_model_step_90000.ckpt-9
# INIT_CHECKPOINT_10=run2/pretrained/bert_model.ckpt-10

INIT_CHECKPOINT_1=run_2020_04_25_1587823510/pretrained/bert_model.ckpt-4
TRAIN_BATCH_SIZE=32
LR=2e-5
EVAL_BATCH_SIZE=8


for FINETUNE_DATA in maternal_vaccine_stance_lshtm covid_worry covid_category twitter_sentiment_semeval vaccine_sentiment_epfl
do
  python run_finetune.py --run_prefix martin_v2 --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP --finetune_data $FINETUNE_DATA --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 0
  python run_finetune.py --run_prefix martin_v2 --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP --finetune_data $FINETUNE_DATA --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 1 --init_checkpoint $INIT_CHECKPOINT_1
done

# optional arguments:
#   -h, --help            show this help message and exit
#   --run_prefix RUN_PREFIX
#                         Prefix to be added to all runs. Useful to group runs
#                         (default: None)
#   --bucket_name BUCKET_NAME
#                         Bucket name (default: cb-tpu-projects)
#   --project_name PROJECT_NAME
#                         Name of subfolder in Google bucket (default: covid-
#                         bert)
#   --finetune_data {maternal_vaccine_stance_lshtm,covid_worry,vaccine_sentiment_epfl,twitter_sentiment_semeval,SST-2,covid_category}
#                         Finetune data folder name. The folder has to be
#                         located in gs://{bucket_name}/{project_name}/finetune/
#                         finetune_data/{finetune_data}. TFrecord files
#                         (train.tfrecord and dev.tfrecord as well as meta.json)
#                         should be located in a subfolder gs://{bucket_name}/{p
#                         roject_name}/finetune/finetune_data/{finetune_data}/tf
#                         record/ (default: maternal_vaccine_stance_lshtm)
#   --tpu_ip TPU_IP       IP-address of the TPU (default: 10.74.219.210)
#   --not_use_tpu         Do not use TPUs (default: False)
#   --num_gpus NUM_GPUS   Number of GPUs to use (default: 1)
#   --init_checkpoint INIT_CHECKPOINT
#                         Run name to initialize checkpoint from. Example:
#                         "run2/ctl_step_8000.ckpt-8". By default using a
#                         pretrained model from
#                         gs://{bucket_name}/pretrained_models/ (default: None)
#   --init_checkpoint_index INIT_CHECKPOINT_INDEX
#                         Checkpoint index. This argument is ignored and only
#                         added for reporting. (default: None)
#   --repeats REPEATS     Number of times the script should run. Default is 1
#                         (default: 1)
#   --num_epochs NUM_EPOCHS
#                         Number of epochs (default: 3)
#   --limit_train_steps LIMIT_TRAIN_STEPS
#                         Limit the number of train steps per epoch. Useful for
#                         testing. (default: None)
#   --limit_eval_steps LIMIT_EVAL_STEPS
#                         Limit the number of eval steps per epoch. Useful for
#                         testing. (default: None)
#   --train_batch_size TRAIN_BATCH_SIZE
#                         Training batch size (default: 32)
#   --eval_batch_size EVAL_BATCH_SIZE
#                         Eval batch size (default: 32)
#   --learning_rate LEARNING_RATE
#                         Learning rate (default: 2e-05)
#   --end_lr END_LR       Final learning rate (default: 0)
#   --warmup_proportion WARMUP_PROPORTION
#                         Learning rate warmup proportion (default: 0.1)
#   --max_seq_length MAX_SEQ_LENGTH
#                         Maximum sequence length (default: 96)
#   --early_stopping_epochs EARLY_STOPPING_EPOCHS
#                         Stop when loss hasn't decreased during n epochs
#                         (default: -1)
#   --model_class {bert}  Model class (default: bert)
#   --model_type {large_uncased,base_uncased}
#                         Model class (default: large_uncased)
#   --optimizer_type {adamw,lamb}
#                         Optimizer (default: adamw)
#   --dtype {fp32,bf16,fp16}
#                         Data type (default: fp32)
#   --steps_per_loop STEPS_PER_LOOP
#                         Steps per loop (default: 10)
#   --time_history_log_steps TIME_HISTORY_LOG_STEPS
#                         Frequency with which to log timing information with
#                         TimeHistory. (default: 10)
#   --model_config_path MODEL_CONFIG_PATH
#                         Path to model config file, by default try to infer
#                         from model_class/model_type args and fetch from
#                         gs://cloud-tpu-checkpoints (default: None)
#
