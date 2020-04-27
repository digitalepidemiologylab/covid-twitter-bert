#!/bin/sh

TPU_IP=10.213.45.170
pretrain_data=v1
# pretrain_data=run_2020_04_27-10-42_1587976935

python run_pretrain.py \
  --pretrain_data $pretrain_data \
  --model_class bert_large_uncased \
  --max_seq_length 96 \
  --max_predictions_per_seq 14 \
  --num_epochs 10 \
  --learning_rate 2e-5 \
  --num_steps_per_epoch 20000 \
  --train_batch_size 1024 \
  --tpu_ip $tpu_ip \
  --steps_per_loop 1000

#   -h, --help            show this help message and exit
#   --tpu_ip tpu_ip       ip-address of the tpu (default: none)
#   --pretrain_data pretrain_data
#                         folder which contains pretrain data. should be located
#                         under gs://{bucket_name}/{project_name}/pretrain/pretr
#                         ain_data/ (default: none)
#   --run_prefix run_prefix
#                         prefix to be added to all runs. useful to group runs
#                         (default: none)
#   --model_class {bert_large_uncased,bert_large_uncased_wwm}
#                         model class to use (default: bert_large_uncased_wwm)
#   --bucket_name bucket_name
#                         bucket name (default: cb-tpu-projects)
#   --project_name project_name
#                         name of subfolder in google bucket (default: covid-
#                         bert)
#   --not_use_tpu         do not use tpus (default: false)
#   --num_gpus num_gpus   number of gpus to use (default: 1)
#   --optimizer_type {adamw,lamb}
#                         optimizer (default: adamw)
#   --train_batch_size train_batch_size
#                         training batch size (default: 32)
#   --num_epochs num_epochs
#                         number of epochs (default: 3)
#   --num_steps_per_epoch num_steps_per_epoch
#                         number of steps per epoch (default: 1000)
#   --warmup_steps warmup_steps
#                         warmup steps (default: 10000)
#   --learning_rate learning_rate
#                         learning rate (default: 2e-05)
#   --end_lr end_lr       final learning rate (default: 0)
#   --max_seq_length max_seq_length
#                         maximum sequence length. sequences longer than this
#                         will be truncated, and sequences shorter than this
#                         will be padded. (default: 96)
#   --max_predictions_per_seq max_predictions_per_seq
#                         maximum predictions per sequence_output. (default: 14)
#   --dtype {fp32,bf16,fp16}
#                         data type (default: fp32)
#   --steps_per_loop steps_per_loop
#                         steps per loop (default: 10)
#   --time_history_log_steps time_history_log_steps
#                         frequency with which to log timing information with
#                         timehistory. (default: 1000)
