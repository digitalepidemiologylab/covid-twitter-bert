import sys
import os
import datetime
import pprint
import uuid
import time
import argparse
import logging
import tqdm
from utils.finetune_helpers import performance_metrics, get_predictions_output
from utils.misc import  append_to_csv, save_to_json


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import modeling
import optimization
import run_classifier
import tokenization

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


##############################
########## CONSTANTS ######### Martin - ALL PATHS NEEDS TO BE CHANGED
##############################
BERT_MODEL_DIR = 'gs://perepublic/multi_cased_L-12_H-768_A-12/'#MOVE TO EXPERIMENT ARGUMENT??
BERT_MODEL_NAME = 'bert_model.ckpt'#MOVE TO EXPERIMENT ARGUMENT??
BERT_MODEL_FILE = os.path.join(BERT_MODEL_DIR, BERT_MODEL_NAME)#MOVE TO EXPERIMENT ARGUMENT??
TEMP_OUTPUT_BASEDIR = 'gs://perepublic/finetuned_models/'
LOG_CSV_DIR = 'log_csv/'
PREDICTIONS_JSON_DIR = 'predictions_json/'
HIDDEN_STATE_JSON_DIR = 'hidden_state_json/'

logdirs = [LOG_CSV_DIR, PREDICTIONS_JSON_DIR, HIDDEN_STATE_JSON_DIR]

for d in logdirs:
    if not os.path.exists(d):
        os.makedirs(d)

##############################
####### HYPERPARAMETERS ######
##############################
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 96
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
WARMUP_PROPORTION = 0.1

##############################
############ CONFIG ##########
##############################
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 100 #Martin - check setting here... maybe 10?
LOWER_CASED = True ##This is currently not used..

##############################
########### FUNCTIONS ########
##############################
def tpu_init(ip):
    #Set up the TPU
    from google.colab import auth
    auth.authenticate_user()
    tpu_address = 'grpc://' + str(ip) + ':8470'

    with tf.Session(tpu_address) as session:
        logger.info('TPU devices:')
        pprint.pprint(session.list_devices())
    logger.info(f'TPU address is active on {tpu_address}')
    return tpu_address

class ClassificationData:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "label"

  def __init__(self, train, dev, test, tokenizer: FullTokenizer, classes, max_seq_len=MAX_SEQ_LENGTH):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.dev_x, self.dev_y), (self.test_x, self.test_y)) = map(self._prepare, [train, dev, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[ClassificationData.DATA_COLUMN], row[ClassificationData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)

def create_model(max_seq_len, bert_ckpt_file):

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
            
    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    ##Martin - Change to logging
    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    
    ##Martin - classed needs to be changed here
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)
            
    return model


##############################
##### DEFINE EXPERIMENTS #####
##############################

experiment_definitions = {
    "1": {
        "name": "covid_worry - base",
        "pretrain_steps": "0",
        "checkpoint": "gs://martins/covid_worry/base",
        "dataset_path": "gs://martins/covid_worry/dataset/"
    },
    "2": {
        "name": "covid_worry - 10000",
        "pretrain_steps": "10000",
        "checkpoint": "gs://martins/covid_worry/checkpoint10000",
        "dataset_path": "gs://martins/covid_worry/dataset/"
    }
}

###########################
##### RUN EXPERIMENTS #####
###########################


def run_experiment(exp_nr, use_tpu, tpu_address, repeat, min_num_epochs, username, comment, store_last_layer):
    logger.info(f'Getting ready to run the following experiments for {repeat} repeats: {experiments}')

    #Martin - fix path
    #I think this has a LOWER CASE parameter check
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))


    ###Martin Needs to be changes to correct paths...
    train = pd.read_csv("train.csv")
    dev = pd.read_csv("dev.csv")
    test = pd.read_csv("test.csv")
    classes = train.intent.unique().tolist()
    data = ClassificationData(train, dev, test, tokenizer, classes, max_seq_len=MAX_SEQ_LENGTH)

    logger.info(f"***** Starting Experiment {exp_nr} *******")
    logger.info(f"***** {experiment_definitions[exp_nr]['name']} ******")
    logger.info("***********************************************")

    #Get a unique ID for every experiment run
    experiment_id = str(uuid.uuid4())

    ###########################
    ######### TRAINING ########
    ###########################

    temp_output_dir = os.path.join(
        TEMP_OUTPUT_BASEDIR,experiment_id)

    os.environ['TFHUB_CACHE_DIR'] = temp_output_dir
    logger.info(f"***** Setting temporary dir {temp_output_dir} **")
    logger.info(f"***** Train started in {temp_output_dir} **")

    if tpu_address:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
    else:
        tpu_cluster_resolver = None

    #Initiation
    model = create_model(data.MAX_SEQ_LENGTH, bert_ckpt_file)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ##Add F1-score here??
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    #Martin - change path
    log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)


    ##Martin - Not sure about epochs here. Do this need to be defined? Can we abort?
    history = model.fit(
        x=data.train_x, 
        y=data.train_y,
        validation_split=0.1,
        batch_size=16,
        shuffle=True,
        epochs=20,
        callbacks=[tensorboard_callback]
    )

    
    ##Martin - from old file - do we need=
    tf.logging.info('  Num steps = %d', num_train_steps)


    ######################################
    ######### #########PREDICTION ########
    ######################################
    
    ## This is currently not OK. Can we evaluate this after every epoch, and not only in the end?
    y_pred_train = model.predict(data.train_x).argmax(axis=-1)
    y_pred_dev = model.predict(data.dev_x).argmax(axis=-1)
    y_pred_test = model.predict(data.test_x).argmax(axis=-1)
    
    ##Martin - Do whatever you need here....
    ##I just left the old code here... None of the input is working
    
    ## PSUDO CODE
    ## * If epochs > min_num_epochs and y_pred_dev_F1 is dropping 
    ## * Abort training
    ## * Calculate y_pred_test Acc/Loss/F1 etc and save that to log
     

    logger.info('Final scores:')
    logger.info(scores)
    logger.info('***** Finished second half of evaluation of {} at {} *****'.
            format(experiment_definitions[exp_nr]["name"],
                    datetime.datetime.now()))

    # write full dev prediction output
    predictions_output = get_predictions_output(experiment_id, guid, probabilities, y_true, label_mapping=label_mapping, dataset='dev')
    save_to_json(predictions_output, os.path.join(PREDICTIONS_JSON_DIR, f'dev_{experiment_id}.json'))

    # Write log to Training Log File
    data = {
        'Experiment_Name': experiment_definitions[exp_nr]["name"],
        'Experiment_Id':experiment_id,
        'Date': format(datetime.datetime.now()),
        'User': username,
        'Model': BERT_MODEL_NAME,
        'Num_Train_Steps': num_train_steps,
        'Train_Annot_Dataset': train_annot_dataset,
        'Eval_Annot_Dataset': eval_annot_dataset,
        'Learning_Rate': LEARNING_RATE,
        'Max_Seq_Length': MAX_SEQ_LENGTH,
        'Eval_Loss': result['eval_loss'],
        'Loss': result['loss'],
        'Comment': comment,
        **scores
    }

    append_to_csv(data, os.path.join(LOG_CSV_DIR,'fulltrainlog.csv'))
    logger.info(f"***** Completed Experiment {exp_nr} *******")



logger.info(f"***** Completed all experiments in {repeat} repeats. We should now clean up all remaining files *****")
for c in completed_train_dirs:
    logger.info("Deleting these directories: ")
    logger.info("gsutil -m rm -r " + c)
    os.system("gsutil -m rm -r " + c)



def parse_args(args):
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tpu_ip',
        dest='tpu_ip',
        default=None,
        help='IP-address of the TPU')
    parser.add_argument(
        '--use_tpu',
        dest='use_tpu',
        action='store_true',
        default=False,
        help='Use TPU. Set to 1 or 0. If set to false, GPU will be used instead')
    parser.add_argument(
        '-u',
        '--username',
        help='Optional. Username is used in the directory name and in the logfile',
        default='Anonymous')
    parser.add_argument(
        '-r',
        '--repeats',
        help='Number of times the script should run. Default is 1',
        default=1,
        type=int)
    parser.add_argument(
        '-e',
        '--experiments',
        type=str,
        help='Experiment numbers to run. Use , to separate individual runs or - to run a sequence of runs.',
        default='1')
    parser.add_argument(
        '-n',
        '--num_train_steps',
        help='Number of train steps. Default is 100',
        default=100,
        type=int)
    parser.add_argument(
        '--store_last_layer',
        action='store_true',
        default=False,
        help='Store last layer of encoder')
    parser.add_argument(
        '--comment',
        help='Optional. Add a Comment to the logfile for internal reference.',
        default='No Comment')
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)

    #Initialise the TPUs if they are used
    if args.use_tpu == 1:
        use_tpu = True
        tpu_address = tpu_init(args.tpu_ip)
        logger.info('Using TPU')
    else:
        use_tpu = False
        tpu_address = None
        logger.info('Using GPU')

    for repeat in range(args.repeats):

        ##Martin - some changes here. I think having the loop running over the experiments are better implementet here than inside the run_experiment. So it should have "exp_nr" instead of "experiments" as input
        ##Num_train_steps should also be changed. It should continue until dropping. Having a minimun for instance 3.
        ##Eval on dev-set and test on test. Add train_steps to output log
        run_experiment(args.experiments, use_tpu, tpu_address, repeat+1, args.num_epochs,
                       args.username, args.comment, args.store_last_layer)
        logger.info(f'*** Completed repeats {repeat + 1}')


if __name__ == "__main__":
    main(sys.argv[1:])
