import os
import logging
import pandas as pd
from utils.analysis_helpers import get_run_logs, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from utils.misc import ArgParseDefault

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


@plot
def main(args):
    df = get_run_logs(pattern=args.run_prefix)
    if len(df) == 0:
        logger.info('No run logs found')
        sys.exit()
    df_pivot = df.pivot(args.y, args.x, 'f1_macro')
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.set_ylim(len(df_pivot)-0.5, -0.5)
    sns.heatmap(df_pivot, ax=ax, annot=True, fmt='.2f', annot_kws={"fontsize":8})
    save_fig(fig, 'heatmap', version=args.version, plot_formats=['png'])

def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='wwm_v2', help='Prefix to plot heatmap')
    parser.add_argument('-y', default='train_batch_size', help='Y-axis column')
    parser.add_argument('-x', default='learning_rate', help='X-axis column')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=1, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
