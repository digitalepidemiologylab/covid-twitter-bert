#!/usr/bin/env sh

TPU_IP=10.250.176.154
NUM_EPOCHS=5
INIT_CHECKPOINT_10=run2/pretrained/bert_model.ckpt-10
INIT_CHECKPOINT_5=run2/pretrained/bert_model_step_50000.ckpt-5


for FINETUNE_DATA in maternal_vaccine_stance_lshtm covid_worry twittter_sentiment_semeval vaccine_sentiment_epfl
do
  python run_finetune.py --tpu_ip $TPU_IP --finetune_data $FINETUNE_DATA --num_epochs $NUM_EPOCHS --init_checkpoint_index 5 --init_checkpoint $INIT_CHECKPOINT_5
done



# help output
#   -h, --help            show this help message and exit
#   --bucket_name BUCKET_NAME
#                         Bucket name (default: cb-tpu-projects)
#   --project_name PROJECT_NAME
#                         Name of subfolder in Google bucket (default: covid-
#                         bert)
#   --finetune_data {maternal_vaccine_stance_lshtm,covid_worry,vaccine_sentiment_epfl,twittter_sentiment_semeval}
#                         Finetune data folder name. The folder has to be
#                         located in gs://{bucket_name}/{project_name}/finetune/
#                         finetune_data/{finetune_data}. TFrecord files
#                         (train.tfrecord and dev.tfrecord as well as meta.json)
#                         should be located in a subfolder gs://{bucket_name}/{p
#                         roject_name}/finetune/finetune_data/{finetune_data}/tf
#                         record/ (default: maternal_vaccine_stance_lshtm)
#   --tpu_ip TPU_IP       IP-address of the TPU (default: 10.217.209.114)
#   --not_use_tpu         Do not use TPUs (default: False)
#   --num_gpus NUM_GPUS   Number of GPUs to use (default: 1)
#   --init_checkpoint INIT_CHECKPOINT
#                         Run name to initialize checkpoint from. Example:
#                         "run2/ctl_step_8000.ckpt-8". By default using a
#                         pretrained model from gs://cloud-tpu-checkpoints.
#                         (default: None)
#   --repeats REPEATS     Number of times the script should run. Default is 1
#                         (default: 1)
#   --num_epochs NUM_EPOCHS
#                         Number of epochs (default: 1)
#   --train_batch_size TRAIN_BATCH_SIZE
#                         Training batch size (default: 32)
#   --eval_batch_size EVAL_BATCH_SIZE
#                         Eval batch size (default: 32)
#   --learning_rate LEARNING_RATE
#                         Learning rate (default: 5e-05)
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
#                         Steps per loop (default: 1000)
#   --log_steps LOG_STEPS
#                         Frequency with which to log timing information with
#                         TimeHistory. (default: 1000)
#   --model_config_path MODEL_CONFIG_PATH
#                         Path to model config file, by default try to infer
#                         from model_class/model_type args and fetch from
#                         gs://cloud-tpu-checkpoints (default: None)
#
