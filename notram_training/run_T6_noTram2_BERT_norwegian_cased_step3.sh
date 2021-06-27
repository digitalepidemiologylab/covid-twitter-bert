PROJECT_NAME=notram_v2
BUCKET_NAME=notram-west4-a
TPU_IP=10.105.231.34
RUN_PREFIX=T4_noTram2_BERT_norwegian_cased
TRAIN_BATCH_SIZE=384
PRETRAIN_DATA=corpus2b_cased_512
MODEL_CLASS=bert_base_norwegian_cased
NUM_EPOCHS=25
MAX_SEQ_LENGTH=512
MAX_PREDICTIONS_PER_SEQ=77
LEARNING_RATE=909e-7
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=100000
WARMUP_STEPS=0
OPTIMIZER_TYPE=lamb
INIT_CHECKPOINT=run_2021-03-03_10-32-24_111182_T4_noTram2_BERT_norwegian_cased/ctl_step_2200000.ckpt-1
LOAD_MLM_NSP_WEIGHTS=True
#EXPECT_PARTIAL=True #Unable to load LAMB optimizer

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
#  --expect_partial $EXPECT_PARTIAL
