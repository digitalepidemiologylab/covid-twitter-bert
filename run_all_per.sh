#!/usr/bin/env sh

TPU_IP=10.213.45.170

NUM_EPOCHS=10

PRETRAIN_RUN_PER=run_per_fixedlr_2e-5__2020_04_26-14-35_1587911750
PRETRAIN_RUN_MARTIN=run_2020_04_26-15-21_1587914486

INIT_CHECKPOINT_3=pretrained/bert_model_step_60000.ckpt-3
INIT_CHECKPOINT_6=pretrained/bert_model_step_1200000.ckpt-6

TRAIN_BATCH_SIZE=32
LR=2e-5
EVAL_BATCH_SIZE=8
FINETUNE_DATA_V1=v1
FINETUNE_DATA_V2=run_2020_04_27-23-57_1588024625



#for FINETUNE_DATASET in maternal_vaccine_stance_lshtm covid_worry covid_category twitter_sentiment_semeval vaccine_sentiment_epfl SST-2
for i in {1..5}
do
  for FINETUNE_DATASET in maternal_vaccine_stance_lshtm
  do
    python run_finetune.py --run_prefix base_lrtest --model_class bert_large_uncased --finetune_data ${FINETUNE_DATA_V1}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP  --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 0

    python run_finetune.py --run_prefix per_lrtest --model_class bert_large_uncased --finetune_data ${FINETUNE_DATA_V1}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP  --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 3 --init_checkpoint ${PRETRAIN_RUN_PER}/$INIT_CHECKPOINT_3

    python run_finetune.py --run_prefix per_lrtest --model_class bert_large_uncased --finetune_data ${FINETUNE_DATA_V1}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP  --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 6 --init_checkpoint ${PRETRAIN_RUN_PER}/$INIT_CHECKPOINT_6

    python run_finetune.py --run_prefix martin_lrtest --model_class bert_large_uncased --finetune_data ${FINETUNE_DATA_V1}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP  --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 3 --init_checkpoint ${PRETRAIN_RUN_MARTIN}/$INIT_CHECKPOINT_3

    python run_finetune.py --run_prefix martin_lrtest --model_class bert_large_uncased --finetune_data ${FINETUNE_DATA_V1}/${FINETUNE_DATASET} --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --tpu_ip $TPU_IP  --num_epochs $NUM_EPOCHS --learning_rate $LR --init_checkpoint_index 6 --init_checkpoint ${PRETRAIN_RUN_MARTIN}/$INIT_CHECKPOINT_6
  done
done

# --finetune_data FINETUNE_DATA
#                       Finetune data folder sub path. Path has to be in gs://
#                       {bucket_name}/{project_name}/finetune/finetune_data/{f
#                       inetune_data}. This folder includes a meta.json
#                       (containing meta info about the dataset), and a file
#                       label_mapping.json. TFrecord files (train.tfrecords
#                       and dev.tfrecords) should be located in a subfolder gs
#                       ://{bucket_name}/{project_name}/finetune/finetune_data
#                       /{finetune_data}/tfrecords/ (default: None)
# --tpu_ip TPU_IP       IP-address of the TPU (default: None)
# --run_prefix RUN_PREFIX
#                       Prefix to be added to all runs. Useful to group runs
#                       (default: None)
# --bucket_name BUCKET_NAME
#                       Bucket name (default: cb-tpu-projects)
# --project_name PROJECT_NAME
#                       Name of subfolder in Google bucket (default: covid-
#                       bert)
# --model_class {bert_large_uncased,bert_large_uncased_wwm}
#                       Model class to use (default: bert_large_uncased_wwm)
# --not_use_tpu         Do not use TPUs (default: False)
# --num_gpus NUM_GPUS   Number of GPUs to use (default: 1)
# --init_checkpoint INIT_CHECKPOINT
#                       Run name to initialize checkpoint from. Example:
#                       "run2/ctl_step_8000.ckpt-8". By default using a
#                       pretrained model from
#                       gs://{bucket_name}/pretrained_models/ (default: None)
# --init_checkpoint_index INIT_CHECKPOINT_INDEX
#                       Checkpoint index. This argument is ignored and only
#                       added for reporting. (default: None)
# --repeats REPEATS     Number of times the script should run. Default is 1
#                       (default: 1)
# --num_epochs NUM_EPOCHS
#                       Number of epochs (default: 3)
# --limit_train_steps LIMIT_TRAIN_STEPS
#                       Limit the number of train steps per epoch. Useful for
#                       testing. (default: None)
# --limit_eval_steps LIMIT_EVAL_STEPS
#                       Limit the number of eval steps per epoch. Useful for
#                       testing. (default: None)
# --train_batch_size TRAIN_BATCH_SIZE
#                       Training batch size (default: 32)
# --eval_batch_size EVAL_BATCH_SIZE
#                       Eval batch size (default: 32)
# --learning_rate LEARNING_RATE
#                       Learning rate (default: 2e-05)
# --end_lr END_LR       Final learning rate (default: 0)
# --warmup_proportion WARMUP_PROPORTION
#                       Learning rate warmup proportion (default: 0.1)
# --max_seq_length MAX_SEQ_LENGTH
#                       Maximum sequence length (default: 96)
# --early_stopping_epochs EARLY_STOPPING_EPOCHS
#                       Stop when loss hasn't decreased during n epochs
#                       (default: -1)
# --optimizer_type {adamw,lamb}
#                       Optimizer (default: adamw)
# --dtype {fp32,bf16,fp16}
#                       Data type (default: fp32)
# --steps_per_loop STEPS_PER_LOOP
#                       Steps per loop (default: 10)
# --time_history_log_steps TIME_HISTORY_LOG_STEPS
#                       Frequency with which to log timing information with
#                       TimeHistory. (default: 10)
# --model_config_path MODEL_CONFIG_PATH
#                       Path to model config file, by default fetch from
#                       PRETRAINED_MODELS["location"] (default: None)