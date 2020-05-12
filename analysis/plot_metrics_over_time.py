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
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


@plot
def main(args):
    df = get_train_logs()
    df = df[df.run_name.str.contains(args.run_prefix)]

    # transform to dict
    df['all_scores'] = df.all_scores.apply(ast.literal_eval)
    _df = pd.DataFrame(df.all_scores.tolist(), index=df.index)
    score_cols = []
    for col in _df:
        score_col = int(col)
        df[score_col] = _df[col].apply(lambda s: s[args.metric] if args.metric in s else np.nan)
        score_cols.append(score_col)
    df = df[score_cols + ['finetune_data', 'init_checkpoint_index']]
    df = df.groupby(['finetune_data', 'init_checkpoint_index']).mean().reset_index()
    df = df.melt(value_vars=score_cols, id_vars=['finetune_data', 'init_checkpoint_index'], var_name='epoch', value_name=args.metric)
    fig = sns.relplot(x='epoch', y=args.metric, hue='init_checkpoint_index', row='finetune_data', kind='line', data=df, height=2, aspect=1.3)
    for ax in fig.axes:
        ax[0].grid()
        ax[0].set_ylim((None, 1))

    # plotting
    save_fig(fig, 'metrics_vs_time', version=args.version, plot_formats=['png'])

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='eval_wwm_v4', help='Prefix to plot heatmap')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=8, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
