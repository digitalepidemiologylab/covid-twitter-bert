#!/bin/sh

TPU_IP=10.74.219.210

python run_pretrain.py \
  --pretrain_data v1 \
  --max_seq_length 96 \
  --max_predictions_per_seq 14 \
  --num_epochs 10 \
  --learning_rate 2e-5 \
  --num_steps_per_epoch 20000 \
  --train_batch_size 1024 \
  --tpu_ip $TPU_IP \
  --steps_per_loop 1000

# optional arguments:
#   -h, --help            show this help message and exit
#   --run_prefix RUN_PREFIX
#                         Prefix to be added to all runs. Useful to group runs
#                         (default: None)
#   --pretrain_data PRETRAIN_DATA
#                         Folder which contains pretrain data. Should be located
#                         under gs://{bucket_name}/{project_name}/pretrain/pretr
#                         ain_data/ (default: v1)
#   --bucket_name BUCKET_NAME
#                         Bucket name (default: cb-tpu-projects)
#   --project_name PROJECT_NAME
#                         Name of subfolder in Google bucket (default: covid-
#                         bert)
#   --tpu_ip TPU_IP       IP-address of the TPU (default: 10.74.219.210)
#   --not_use_tpu         Do not use TPUs (default: False)
#   --num_gpus NUM_GPUS   Number of GPUs to use (default: 1)
#   --optimizer_type {adamw,lamb}
#                         Optimizer (default: adamw)
#   --train_batch_size TRAIN_BATCH_SIZE
#                         Training batch size (default: 32)
#   --num_epochs NUM_EPOCHS
#                         Number of epochs (default: 3)
#   --num_steps_per_epoch NUM_STEPS_PER_EPOCH
#                         Number of steps per epoch (default: 1000)
#   --warmup_steps WARMUP_STEPS
#                         Warmup steps (default: 10000)
#   --learning_rate LEARNING_RATE
#                         Learning rate (default: 2e-05)
#   --end_lr END_LR       Final learning rate (default: 0)
#   --max_seq_length MAX_SEQ_LENGTH
#                         Maximum sequence length. Sequences longer than this
#                         will be truncated, and sequences shorter than this
#                         will be padded. (default: 96)
#   --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
#                         Maximum predictions per sequence_output. (default: 14)
#   --dtype {fp32,bf16,fp16}
#                         Data type (default: fp32)
#   --model_class {bert}  Model class (default: bert)
#   --model_type {large_uncased,base_uncased}
#                         Model class to pretraining (default: large_uncased)
#   --steps_per_loop STEPS_PER_LOOP
#                         Steps per loop (default: 10)

