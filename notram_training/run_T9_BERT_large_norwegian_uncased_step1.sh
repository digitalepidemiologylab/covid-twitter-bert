PROJECT_NAME=notram_v2
BUCKET_NAME=notram-east1-d
#TPU_IP=10.36.255.114
TPU_NAME=nbpod-east
RUN_PREFIX=T9_BERT_large_norwegian_uncased
#TRAIN_BATCH_SIZE=8192
TRAIN_BATCH_SIZE=32000
PRETRAIN_DATA=corpus2b_uncased_128
MODEL_CLASS=bert_base_norwegian_uncased
NUM_EPOCHS=10
MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=19
#LEARNING_RATE=25e-4
#END_LEARNING_RATE=9e-4
LEARNING_RATE=50e-4
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=10000
WARMUP_STEPS=10000
OPTIMIZER_TYPE=lamb
INIT_WEIGHTS=True

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
  --init_weights $INIT_WEIGHTS
