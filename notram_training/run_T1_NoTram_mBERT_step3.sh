PROJECT_NAME=notram_v1
BUCKET_NAME=notram-west4-a
TPU_IP=10.170.222.114
RUN_PREFIX=T1_NoTram_mBERT_step3
TRAIN_BATCH_SIZE=384
PRETRAIN_DATA=corpus1_512
MODEL_CLASS=bert_multi_cased
NUM_EPOCHS=12
MAX_SEQ_LENGTH=512
MAX_PREDICTIONS_PER_SEQ=77
LEARNING_RATE=24e-4
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=100000
WARMUP_STEPS=0
OPTIMIZER_TYPE=lamb
INIT_CHECKPOINT=run_2020-12-13_11-33-27_046420_T1_NoTram_mBERT_step2/ctl_step_1000000.ckpt-3
LOAD_MLM_NSP_WEIGHTS=True
EXPECT_PARTIAL=True

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
  --load_mlm_nsp_weights $LOAD_MLM_NSP_WEIGHTS \
  --expect_partial $EXPECT_PARTIAL

