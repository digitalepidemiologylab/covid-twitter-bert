import os
import logging
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
    df = get_run_logs(pattern=args.run_prefix, bucket_name=args.bucket_name, project_name=args.project_name)
    if len(df) == 0:
        logger.info('No run logs found')
        sys.exit()
    df.loc[(df.model_class == 'covid-twitter-bert'), 'init_checkpoint'] = 'ct-bert-v1/bla'
    df.loc[(df.model_class != 'covid-twitter-bert') & (df.init_checkpoint_index == 0), 'init_checkpoint'] = 'bert-large-uncased-wwm/bla'
    df = df.reset_index(drop=True)
    df.finetune_data = df.finetune_data.apply(lambda s: s.split('/')[-1])
    df['exp'] = df.init_checkpoint.apply(lambda s: s.split('/')[0])
    df.loc[df.exp.str.contains('B5'), 'exp'] = 'expB5_34e5_nodecay_concat'
    df = df[['run_name', 'exp', 'finetune_data', 'init_checkpoint_index', args.metric]]

    # exclude single checkpoint runs
    df = df[~df.exp.isin(['ct-bert-v1', 'bert-large-uncased-wwm'])]

    g = sns.FacetGrid(df, row='finetune_data', hue='exp', aspect=3, height=2, sharey=False)
    g = g.map(sns.lineplot, 'init_checkpoint_index', args.metric)
    g.add_legend()
    # plotting
    save_fig(plt.gcf(), 'metrics_by_checkpoint', version=args.version, plot_formats=['png'])

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-us-central1', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert-v2', help='Project name')
    parser.add_argument('--run_prefix', default='(exp2|exp3|exp4|B3|B4|B5)', help='Run prefix')
    parser.add_argument('--confidence_intervals', default='ci', choices=['ci', 'sd'], help='Type of confidence intervals')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=14, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
