PROJECT_NAME=notram_v2
BUCKET_NAME=notram-west4-a
#TPU_IP=10.36.255.114
TPU_NAME=nbpod4
RUN_PREFIX=T6POD_BERT_base_norwegian_cased
TRAIN_BATCH_SIZE=32000
PRETRAIN_DATA=corpus2b_cased_128
MODEL_CLASS=bert_base_norwegian_cased
NUM_EPOCHS=14
MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=19
LEARNING_RATE=24e-4
END_LEARNING_RATE=24e-4
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=10000
WARMUP_STEPS=5000
OPTIMIZER_TYPE=lamb
INIT_CHECKPOINT=run_2021-03-23_08-31-39_269081_T6POD_BERT_base_norwegian_cased/ctl_step_50000.ckpt-5
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
