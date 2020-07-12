import os
import sys
import logging
import pandas as pd
from utils.analysis_helpers import get_run_logs, save_fig, plot
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.misc import ArgParseDefault
from collections import defaultdict
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

@plot
def main(args):
    df = get_run_logs(pattern=args.run_prefix, project_name=args.project_name, bucket_name=args.bucket_name)
    # due to multiple runs that have been run with the same prefix
    if len(df) == 0:
        logger.info('No run logs found')
        sys.exit()
    # df.loc[(df.model_class != 'covid-twitter-bert') & (df.init_checkpoint_index == 0), 'init_checkpoint'] = 'bert-large-uncased-wwm'
    # df = df[df['init_checkpoint'] != 'bert-large-uncased-wwm']
    df = df.reset_index(drop=True)
    # make run names nicer
    df.finetune_data = df.finetune_data.apply(lambda s: s.split('/')[-1])
    # ['SemEval2016_6_atheism', 'SemEval2016_4a_sentiment', 'SemEval2016_6_climate_change_is_a_real_concern', 'SemEval2016_6_feminist_movement', 'SemEval2016_6_hillary_clinton', 'SemEval2016_6_legalization_of_abortion', 'SemEval2016_6_sentiment', 'SemEval2017_4a_sentiment', 'SemEval2018_3a_irony', 'covid_category', 'SST-2']
    df = df[~df.finetune_data.isin(
        ['SemEval2016_6_climate_change_is_a_real_concern', 'SemEval2016_6_feminist_movement', 'SemEval2016_6_hillary_clinton', 'SemEval2016_6_legalization_of_abortion']
        )]
    df = df[['init_checkpoint_index', args.metric, 'init_checkpoint', 'model_class', 'finetune_data']]
    mean_ct_bert_v1 = df.loc[df.model_class == 'covid-twitter-bert', args.metric].mean()
    mean_bert_large = df.loc[(df.model_class == 'bert_large_uncased_wwm') & (df.init_checkpoint_index == 0), args.metric].mean()
    df = df.dropna(subset=['init_checkpoint'])
    df['exp'] = df.init_checkpoint.apply(lambda s: s.split('/')[0])
    # df['exp'] = df.init_checkpoint.apply(get_exp)

    # for finetune_data, grp in df.groupby('finetune_data'):
    #     mean_base = grp[grp.init_checkpoint_index == 0].mean()[args.metric]
    #     df.loc[df.finetune_data == finetune_data, 'metric_potential'] = (df.loc[df.finetune_data == finetune_data, args.metric] - mean_base)/(1 - mean_base)

    # plotting
    height = 2.6
    width = 1.61803398875 * height
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    for (finetune_data, init_checkpoint_index, exp), grp in df.groupby(['finetune_data', 'init_checkpoint_index', 'exp']):
        df.loc[(df.init_checkpoint_index == init_checkpoint_index) & (df.exp == exp) & (df.finetune_data == finetune_data), args.metric] = grp[args.metric].median()

    sns.lineplot(x='init_checkpoint_index', y=args.metric, hue='exp', data=df, ax=ax)
    ax.axhline(y=mean_ct_bert_v1, ls ='--', c='k')
    ax.axhline(y=mean_bert_large, ls ='--', c='red')
    # df.groupby('init_checkpoint_index').mean()[args.metric].plot(ax=ax, color='k', ls='dashed', label='Average')
    ax.grid()
    ax.set_ylabel(f'{args.metric}')
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., frameon=False, title=False)
    # legend.texts[0].set_text("Evaluation dataset")
    ax.set_xlabel('Pretraining step')

    # plotting
    save_fig(plt.gcf(), 'fig2', version=args.version, plot_formats=['png', 'pdf'])


def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-us-central1', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert-v2', help='Project name')
    parser.add_argument('--run_prefix', default='ct_bert_v2_eval', help='Run prefix')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=6, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
