import os
import logging
import pandas as pd
from utils.analysis_helpers import get_train_logs, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from utils.misc import ArgParseDefault
from matplotlib.ticker import MaxNLocator
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


@plot
def main(args):
    df = get_train_logs()
    df = df[df.run_name.str.contains(args.run_prefix)]
    data = {}
    for i, row in df.iterrows():
        scores = ast.literal_eval(row.all_scores)
        data[row.run_name] = {}
        for epoch, score in enumerate(scores):
            data[row.run_name][epoch + 1] = score[args.metric]

    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    for run_name, scores in data.items():
        rec = df[df['run_name'] == run_name].iloc[0]
        label = f'bs: {rec.train_batch_size}, lr: {rec.learning_rate}'
        ax.plot(list(scores.keys()), list(scores.values()), label=label)
    ax.set_ylim([0, 0.8])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(args.metric)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()

    # plotting
    save_fig(fig, 'metrics_vs_time', version=args.version, plot_formats=['png'])

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='v1', help='Prefix to plot heatmap')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=3, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
