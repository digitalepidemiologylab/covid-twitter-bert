import os
import sys
from utils.analysis_helpers import get_run_logs, save_fig, plot, get_summary_logs
import matplotlib.pyplot as plt
import seaborn as sns
import logging
sys.path.append('..')
from utils.misc import ArgParseDefault

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    df = get_summary_logs(pattern=args.run_prefix, dataset_type=args.dataset_type, bucket_name=args.bucket_name, project_name=args.project_name)
    if len(df) == 0:
        logger.info('No runs found')
    df_run = get_run_logs(pattern=args.run_prefix, run_type='pretrain', bucket_name=args.bucket_name, project_name=args.project_name)
    df.index.name = 'step'
    df = df.reset_index()
    df = df.merge(df_run, on='run_name', how='inner')
    df = df[df.step > 8000]
    df['train_examples'] = df.step * df.train_batch_size
    df.loc[df.run_name.str.contains('B5'), 'run_name'] = 'expB5_34e5_nodecay_concat'
    df = df[['run_name', 'train_examples', 'step', *args.metrics]]
    df = df.melt(id_vars=['run_name', 'train_examples', 'step'], var_name='metric')

    df = df[df.metric.isin(args.metrics)]
    g = sns.FacetGrid(df, row='metric', hue='run_name', sharey=False, aspect=3, height=2)
    g = g.map(sns.lineplot, 'train_examples', 'value')
    for axes in g.axes:
        for ax in axes:
            ax.grid()
    g.add_legend()
    fig = plt.gcf()
    fig.suptitle(f'Pretrain {args.dataset_type} metrics')
    fig.subplots_adjust(top=0.9)
    save_fig(fig, 'pretrain_metrics', version=args.version)

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-us-central1', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert-v2', help='Project name')
    parser.add_argument('--run_prefix', default='(exp2|exp3|exp4|B3|B4|B5)', help='Run prefix')
    parser.add_argument('--metrics', default=['next_sentence_loss', 'next_sentence_accuracy', 'masked_lm_accuracy', 'lm_example_loss'], help='Metrics to plot')
    parser.add_argument('--dataset_type', default='eval', choices=['train', 'eval'], help='Pretrain dataset type')
    parser.add_argument('-v', '--version', type=int, default=8, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
