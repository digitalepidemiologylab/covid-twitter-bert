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

    # filter by highest checkpoint_index
    if args.filter_by_highest_checkpoint:
        df['use'] = False
        for exp, grp in df.groupby('exp'):
            max_checkpoint_index = grp.init_checkpoint_index.max()
            if pd.isna(max_checkpoint_index):
                df.loc[(df.exp == exp), 'use'] = True
            else:
                df.loc[(df.exp == exp) & (df.init_checkpoint_index == max_checkpoint_index), 'use'] = True
        df = df[df.use]

    if args.normalized_mean:
        for (finetune_data, exp), grp in df.groupby(['finetune_data', 'exp']):
            if exp != args.normalized_mean_against:
                current_scores  = df.loc[(df.exp == exp) & (df.finetune_data == finetune_data), args.metric]
                mean_bert = df.loc[(df.exp == args.normalized_mean_against) & (df.finetune_data == finetune_data), args.metric].mean()
                df.loc[(df.exp == exp) & (df.finetune_data == finetune_data), args.metric] = (current_scores - mean_bert)/mean_bert
        df = df[df.exp != args.normalized_mean_against]
        y_label = f'Normalized mean {args.metric} improvement vs. {args.normalized_mean_against}'
    else:
        y_label = f'{args.metric}'
    fig = sns.catplot(y=args.metric, x='finetune_data', hue=args.hue, data=df, kind='bar', height=4, aspect=1.5)
    plt.gca().set_ylabel(y_label)
    fig.set_xticklabels(rotation=90)
    plt.gca().set_title(args.run_prefix + f' (confidence intervals: {args.confidence_intervals})')

    # plotting
    save_fig(fig, 'metrics_by_checkpoint', version=args.version, plot_formats=['png'])

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-us-central1', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert-v2', help='Project name')
    parser.add_argument('--run_prefix', default='B5', help='Run prefix')
    parser.add_argument('--hue', default='init_checkpoint_index', help='Column type to compare')
    parser.add_argument('--filter_by_highest_checkpoint', type=bool, default=False, help='Only use highest checkpoint for any experiment')
    parser.add_argument('--normalized_mean', type=bool, default=False, help='Plot normalized mean')
    parser.add_argument('--normalized_mean_against', type=str, default='bert-large-uncased-wwm', help='Plot normalized mean against this experiment')
    parser.add_argument('--confidence_intervals', default='ci', choices=['ci', 'sd'], help='Type of confidence intervals')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=12, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
