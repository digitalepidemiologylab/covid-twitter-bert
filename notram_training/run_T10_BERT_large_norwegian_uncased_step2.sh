PROJECT_NAME=notram_v2
BUCKET_NAME=notram-east1-d
#TPU_IP=10.36.255.114
TPU_NAME=nbpod-east
RUN_PREFIX=T10_BERT_large_norwegian_uncased_from_180000_no_reset
TRAIN_BATCH_SIZE=4096
#TRAIN_BATCH_SIZE=32000
PRETRAIN_DATA=corpus2b_uncased_512
MODEL_CLASS=bert_large_norwegian_uncased
NUM_EPOCHS=36 #Slightly reduced from 40 since we are recovering from 18 instead of 20 to be sure
#We are training (20*10000*32000) = 6.4B examples in the 128. This means a total of 7.11B examples, where 711M should be in 512seq. To be able to do this, we train for 17,3*10000*4096 in 512seq. We add an additional 15% warmup and get to 20 epochs. 
MAX_SEQ_LENGTH=512
MAX_PREDICTIONS_PER_SEQ=77
LEARNING_RATE=4e-4
#17e-4 #Reducing to 4e-4 since that is where thing goes south 
#Set according to 76 minutes article. For it to hit 17e-4 at step 20
#END_LEARNING_RATE=9e-4
#LEARNING_RATE=50e-4
END_LEARNING_RATE=0
STEPS_PER_LOOP=100
NUM_STEPS_PER_EPOCH=10000
WARMUP_STEPS=20000
OPTIMIZER_TYPE=lamb
#INIT_WEIGHTS=True
INIT_CHECKPOINT=run_2021-04-16_17-16-00_628307_T10_BERT_large_norwegian_uncased_from_180000_no_reset/ctl_step_340000.ckpt-5
#INIT_CHECKPOINT=run_2021-04-15_21-19-45_853463_T10_BERT_large_norwegian_uncased_from_180000_no_reset/ctl_step_290000.ckpt-2
#INIT_CHECKPOINT=run_2021-04-13_15-25-59_460783_T10_BERT_large_norwegian_uncased_from_180000_no_reset/ctl_step_270000.ckpt-4 
#INIT_CHECKPOINT=run_2021-04-11_17-39-34_919697_T10_BERT_large_norwegian_uncased_from_180000_no_reset/ctl_step_230000.ckpt-5
#INIT_CHECKPOINT=run_2021-04-08_17-17-22_118288_T10_BERT_large_norwegian_uncased/ctl_step_180000.ckpt-1
#run_2021-03-30_21-16-30_960286_T9_BERT_large_norwegian_uncased/ctl_step_70000.ckpt-2
LOAD_MLM_NSP_WEIGHTS=True
#SET_TRAINSTEP=0

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
#  --set_trainstep $SET_TRAINSTEP \
#  --reset_optimizer
 # --init_weights $INIT_WEIGHTS
