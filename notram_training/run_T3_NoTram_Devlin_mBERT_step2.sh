PROJECT_NAME=notram_v1
BUCKET_NAME=notram-west4-a
TPU_IP=10.109.182.242
RUN_PREFIX=T2_NoTram_Devlin_mBERT_step2
MODEL_CLASS=bert_multi_cased
TRAIN_BATCH_SIZE=256
PRETRAIN_DATA=corpus1_512
NUM_EPOCHS=10
MAX_SEQ_LENGTH=512
MAX_PREDICTIONS_PER_SEQ=77
LEARNING_RATE=2e-5
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=100000
WARMUP_STEPS=0
OPTIMIZER_TYPE=adamw
INIT_CHECKPOINT=run_2020-12-04_15-03-22_229120_T3_NoTram_Devlin_mBERT_step1/ctl_step_900000.ckpt-9
LOAD_MLM_NSP_WEIGHTS=True

python run_pretrain.py \
  --run_prefix $RUN_PREFIX \
  --project_name $PROJECT_NAME \
  --bucket_name $BUCKET_NAME \
  --tpu_ip $TPU_IP \
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
  --load_mlm_nsp_weights $LOAD_MLM_NSP_WEIGHTS
