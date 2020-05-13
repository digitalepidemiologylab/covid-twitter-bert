import os
import sys
import logging
import pandas as pd
from utils.analysis_helpers import find_project_root, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.misc import ArgParseDefault
import json
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

@plot
def main(args):
    folder = os.path.join(find_project_root(), 'data', 'cb-tpu-projects', 'covid-bert', 'pretrain', 'run_2020-04-29_11-14-08_711153_wwm_v2')
    data = {}
    for metric in ['loss', 'masked_lm_accuracy', 'next_sentence_accuracy', 'next_sentence_loss']:
        with open(os.path.join(folder, f'{metric}.json'), 'r') as f:
            d = json.load(f)
            d = {i[1]: i[2] for i in d}
            data[metric] = d
    df = pd.DataFrame(data)
    # plot
    height = 1.8
    width = 1.61803398875 * height
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, height))

    for ax_loss, (loss, acc), (loss_label, acc_label) in zip(axes,
            [['loss', 'masked_lm_accuracy'], ['next_sentence_loss', 'next_sentence_accuracy']],
            [['MLM loss', 'MLM accuracy'], ['NSP loss', 'NSP accuracy']],
            ):
        ax_acc = ax_loss.twinx()
        df[loss].plot(ax=ax_loss, c='C0', label=loss_label)
        df[acc].plot(ax=ax_acc, c='C1', label=acc_label)
        ax_loss.set_ylabel(loss_label, color='C0')
        ax_acc.set_ylabel(acc_label, color='C1')
        ax_loss.set_xlabel('Pretraining step')
        if acc_label == 'NSP accuracy':
            ax_acc.yaxis.set_ticks(np.arange(0.9, 1, .025))
            ax_loss.yaxis.set_ticks(np.arange(0.0, .4, .2))
            ax_acc.grid()
            ax_loss.tick_params(axis='x', which='major', pad=5)
        elif acc_label == 'MLM accuracy':
            ax_acc.yaxis.set_ticks(np.arange(0.65, .725, .025))
            ax_loss.yaxis.set_ticks(np.arange(1.5, 2, 0.2))
            ax_loss.grid()
    # save
    save_fig(fig, 'fig1', version=args.version, plot_formats=['png'])


def parse_args():
    # Parse commandline
    parser = ArgParseDefault()
    parser.add_argument('--run_prefix', default='wwm_v2', help='Prefix to plot heatmap')
    parser.add_argument('-v', '--version', type=int, default=2, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
