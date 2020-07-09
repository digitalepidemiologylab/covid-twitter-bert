PRETRAIN_DATA=run_2020-06-16_22-20-47_002555_wwm_v2
RUN_PREFIX=expB3_24e5_nodecay
BUCKET_NAME=cb-tpu-west4
TPU_NAME=tpu-west4-v3-128-ondemand
MODEL_CLASS=bert_large_uncased_wwm
TRAIN_BATCH_SIZE=16384
EVAL_BATCH_SIZE=16384
PROJECT_NAME=covid-bert-v2


python run_pretrain.py \
  --run_prefix $RUN_PREFIX \
  --project_name $PROJECT_NAME \
  --bucket_name $BUCKET_NAME \
  --tpu_name $TPU_NAME \
  --pretrain_data $PRETRAIN_DATA \
  --model_class $MODEL_CLASS \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --num_epochs 20 \
  --max_seq_length 96 \
  --learning_rate 24e-5 \
  --steps_per_loop 1000 \
  --num_steps_per_epoch 7812 \
  --warmup_steps 10000 \
  --optimizer_type lamb \
  --end_lr 24e-5 \
  --do_eval
