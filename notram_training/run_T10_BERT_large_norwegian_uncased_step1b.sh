PROJECT_NAME=notram_v2
BUCKET_NAME=notram-east1-d
#TPU_IP=10.36.255.114
TPU_NAME=nbpod-east
RUN_PREFIX=T10_BERT_large_norwegian_uncased
#TRAIN_BATCH_SIZE=8192
TRAIN_BATCH_SIZE=32000
PRETRAIN_DATA=corpus2b_uncased_128
MODEL_CLASS=bert_large_norwegian_uncased
NUM_EPOCHS=20
MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=19
#LEARNING_RATE=25e-4
#END_LEARNING_RATE=9e-4
LEARNING_RATE=25e-4
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=10000
WARMUP_STEPS=10000
OPTIMIZER_TYPE=lamb
#INIT_CHECKPOINT=run_2021-04-03_06-25-25_024568_T10_BERT_large_norwegian_uncased/ctl_step_40000.ckpt-4
#INIT_CHECKPOINT=run_2021-04-04_11-57-49_059605_T10_BERT_large_norwegian_uncased/ctl_step_70000.ckpt-3
#iINIT_CHECKPOINT=run_2021-04-05_11-55-03_461702_T10_BERT_large_norwegian_uncased/ctl_step_90000.ckpt-2
#INIT_CHECKPOINT=run_2021-04-06_11-23-29_647695_T10_BERT_large_norwegian_uncased/ctl_step_110000.ckpt-2
#INIT_CHECKPOINT=run_2021-04-07_04-44-30_393761_T10_BERT_large_norwegian_uncased/ctl_step_130000.ckpt-2
INIT_CHECKPOINT=run_2021-04-07_17-11-47_473113_T10_BERT_large_norwegian_uncased/ctl_step_170000.ckpt-4

LOAD_MLM_NSP_WEIGHTS=True


python run_pretrain.py \
  --run_prefix $RUN_PREFIX \
  --project_name $PROJECT_NAME \
  --bucket_name $BUCKET_NAME \
  --tpu_name $TPU_NAME \
  --pretrain_data $PRETRAIN_DATA \
  --model_class $MODEL_CLASS \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
  --learning_rate $LEARNING_RATE \
  --end_lr $END_LEARNING_RATE \
  --steps_per_loop $STEPS_PER_LOOP \
  --num_steps_per_epoch $NUM_STEPS_PER_EPOCH \
  --warmup_steps $WARMUP_STEPS \
  --optimizer_type $OPTIMIZER_TYPE \
  --init_checkpoint $INIT_CHECKPOINT \
  --load_mlm_nsp_weights $LOAD_MLM_NSP_WEIGHTS \
