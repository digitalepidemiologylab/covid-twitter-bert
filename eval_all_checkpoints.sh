#!/usr/bin/env sh

PRETRAIN_RUN=run_2020-04-29_11-14-08_711153_wwm_v2/pretrained
TRAIN_BATCH_SIZE=32
LR=2e-5
EVAL_BATCH_SIZE=8
FINETUNE_DATA=run_2020-04-29_22-20-35_981382
MODEL_CLASS=bert_large_uncased_wwm
NUM_REPEATS=5

declare -A num_epochs_by_dataset
num_epochs_by_dataset=( ["maternal_vaccine_stance_lshtm"]=10 ["covid_worry"]=3 ["covid_category"]=3 ["twitter_sentiment_semeval"]=3 ["vaccine_sentiment_epfl"]=5 ["SST-2"]=3 )

INIT_CHECKPOINT_04=bert_model_step_100000.ckpt-4
INIT_CHECKPOINT_08=bert_model_step_200000.ckpt-8
INIT_CHECKPOINT_12=bert_model_step_300000.ckpt-12
INIT_CHECKPOINT_16=bert_model_step_400000.ckpt-16
INIT_CHECKPOINT_20=bert_model_step_500000.ckpt-20

TPU_1=10.120.173.242
TPU_2=10.245.46.146
TPU_3=10.214.157.194
TPU_4=10.94.160.178
TPU_5=10.236.51.82
TPU_6=10.125.37.218

for FINETUNE_DATASET in maternal_vaccine_stance_lshtm covid_worry covid_category twitter_sentiment_semeval vaccine_sentiment_epfl SST-2
do
  NUM_EPOCHS=${num_epochs_by_dataset[$FINETUNE_DATASET]}
  for i in $(seq 1 $NUM_REPEATS)
  do
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_1 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 0 &
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_2 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 4 --init_checkpoint ${PRETRAIN_RUN}/${INIT_CHECKPOINT_04} &
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_3 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 8 --init_checkpoint ${PRETRAIN_RUN}/${INIT_CHECKPOINT_08} &
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_4 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 12 --init_checkpoint ${PRETRAIN_RUN}/${INIT_CHECKPOINT_12} &
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_5 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 16 --init_checkpoint ${PRETRAIN_RUN}/${INIT_CHECKPOINT_16} &
    python run_finetune_tf21.py --run_prefix eval_wwm_v4 --bucket_name cb-tpu-projects-us --tpu_ip $TPU_6 --model_class $MODEL_CLASS --finetune_data ${FINETUNE_DATA}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 20 --init_checkpoint ${PRETRAIN_RUN}/${INIT_CHECKPOINT_20} &
    wait
  done
done

# optional arguments:
#   -h, --help            show this help message and exit
#   --finetune_data FINETUNE_DATA
#                         Finetune data folder sub path. Path has to be in gs://
#                         {bucket_name}/{project_name}/finetune/finetune_data/{f
#                         inetune_data}. This folder includes a meta.json
#                         (containing meta info about the dataset), and a file
#                         label_mapping.json. TFrecord files (train.tfrecords
#                         and dev.tfrecords) should be located in a subfolder gs
#                         ://{bucket_name}/{project_name}/finetune/finetune_data
#                         /{finetune_data}/tfrecords/ (default: None)
#   --tpu_ip TPU_IP       IP-address of the TPU (default: None)
#   --preemptible_tpu     Dynamically create preemptible TPU (this requires you
#                         to have glcoud installed with suitable permissions)
#                         (default: False)
#   --preemptible_tpu_zone PREEMPTIBLE_TPU_ZONE
#                         Preemptible TPU zone (only if --preemptible_tpu flag
#                         is provided) (default: us-central1-f)
#   --preemptible_tpu_name PREEMPTIBLE_TPU_NAME
#                         Preemptible TPU name (only if --preemptible_tpu flag
#                         is provided) (default: None)
#   --preemptible_tpu_version {nightly,2.1}
#                         Preemptible TPU version (only if --preemptible_tpu
#                         flag is provided) (default: 2.1)
#   --run_prefix RUN_PREFIX
#                         Prefix to be added to all runs. Useful to group runs
#                         (default: None)
#   --bucket_name BUCKET_NAME
#                         Bucket name (default: cb-tpu-projects)
#   --project_name PROJECT_NAME
#                         Name of subfolder in Google bucket (default: covid-
#                         bert)
#   --model_class {bert_large_uncased,bert_large_uncased_wwm}
#                         Model class to use (default: bert_large_uncased_wwm)
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
#                         Path to model config file, by default fetch from
#                         PRETRAINED_MODELS["location"] (default: None)
#   --use_tpu             Use TPU (default: True)
#   --do_not_use_tpu
