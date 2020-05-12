import os
import logging
import sklearn.metrics
import pandas as pd
from utils.analysis_helpers import get_run_logs, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from utils.misc import ArgParseDefault
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


@plot
def main(args):
    raise NotImplementedError
    df = get_run_logs(pattern=args.run_name)
    # label_mapping = get_label_mapping(f_path)
    # labels = list(label_mapping.keys())
    cnf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction)
    df = pd.DataFrame(cnf_matrix, columns=labels, index=labels)
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    sns.heatmap(df, ax=ax, annot=True, fmt='d', annot_kws={"fontsize":8})
    ax.set(xlabel='predicted label', ylabel='true label')
    save_fig(fig, f_path, 'confusion_matrix')


def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_name', required=True, help='Run name to plot confusion matrix for')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
