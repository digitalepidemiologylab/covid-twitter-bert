import os
import logging
import sklearn.metrics
import pandas as pd
from utils.analysis_helpers import find_project_root, save_fig
import seaborn as sns
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from utils.misc import ArgParseDefault

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def get_train_logs():
    f_path = os.path.join(find_project_root(), 'traininglog.csv')
    return pd.read_csv(f_path)


def main(args):
    df = get_train_logs()
    df = df[df.run_name.str.contains(args.run_prefix)]
    df_pivot = df.pivot('train_batch_size', 'learning_rate', 'f1_macro')
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    sns.heatmap(df_pivot, ax=ax, annot=True, fmt='.1f', annot_kws={"fontsize":8})
    save_fig(fig, 'heatmap')



def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='martin_hyperparameters', help='Prefix to plot heatmap')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
