import os
import logging
import pandas as pd
from utils.analysis_helpers import get_train_logs, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from utils.misc import ArgParseDefault
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

@plot
def main(args):
    df = get_train_logs()
    df = df[df.run_name.str.contains(args.run_prefix)]
    fig = sns.catplot(x='finetune_data', y='f1_macro', hue='init_checkpoint_index', data=df, kind='bar', height=4, aspect=1.5)
    fig.set_xticklabels(rotation=45)
    fig.set(ylim=(None, 1))

    # plotting
    save_fig(fig, 'metrics_by_checkpoint', version=args.version, plot_formats=['png'])

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='v1', help='Prefix to plot heatmap')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=2, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
