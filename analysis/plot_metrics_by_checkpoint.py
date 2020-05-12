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
    fig = sns.catplot(x='finetune_data', y=args.metric, hue='init_checkpoint_index', data=df, kind='violin', height=4, aspect=1.5)
    fig.set_xticklabels(rotation=45)
    fig.set(ylim=(0.5, 1))
    plt.gca().set_title(args.run_prefix)

    # plotting
    save_fig(fig, 'metrics_by_checkpoint', version=args.version, plot_formats=['png'])

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='eval_wwm_v4', help='Prefix to plot heatmap')
    parser.add_argument('--metric', default='accuracy', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=10, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
