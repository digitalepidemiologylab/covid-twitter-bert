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
    # df.loc[df.init_checkpoint_index.isna(), 'init_checkpoint'] = 'ct-bert-v1'
logger = logging.getLogger(__name__)

@plot
def main(args):
    df = get_run_logs(pattern=args.run_prefix, bucket_name=args.bucket_name, project_name=args.project_name)
    df.loc[(df.model_class == 'covid-twitter-bert'), 'init_checkpoint'] = 'ct-bert-v1/bla'
    df.loc[(df.model_class != 'covid-twitter-bert') & (df.init_checkpoint_index == 0), 'init_checkpoint'] = 'bert-large-uncased-wwm/bla'
    df = df.reset_index(drop=True)
    df = df[df.init_checkpoint_index.isin([None, 0, 9])]
    n = 3
    to_delete = []
    for (finetune_data, init_checkpoint), grp in df.groupby(['finetune_data', 'init_checkpoint']):
        sorted_vals = grp[args.metric].sort_values(ascending=False)
        to_delete.extend(sorted_vals[n:].index.tolist())

    df = df.drop(index=to_delete)
    df.finetune_data = df.finetune_data.apply(lambda s: s.split('/')[-1])
    df = df[~df.finetune_data.isin(
        ['SemEval2016_6_climate_change_is_a_real_concern', 'SemEval2016_6_feminist_movement', 'SemEval2016_6_hillary_clinton', 'SemEval2016_6_legalization_of_abortion']
        )]
    df['exp'] = df.init_checkpoint.apply(lambda s: s.split('/')[0])
    if len(df) == 0:
        logger.info('No run logs found')
        sys.exit()
    fig = sns.catplot(y=args.metric, x='finetune_data', hue='exp', data=df, kind='bar', height=4, aspect=1.5, ci='sd')
    # plt.legend(loc='bottom')
    # fig.set_xticklabels(rotation=90)
    fig.set(ylim=(0.5, 1))
    plt.gca().set_title(args.run_prefix)

    # plotting
    save_fig(fig, 'metrics_by_checkpoint', version=args.version, plot_formats=['png'])

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--bucket_name', default='cb-tpu-us-central1', help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert-v2', help='Project name')
    parser.add_argument('--run_prefix', default='ct_bert_v2_eval', help='Run prefix')
    parser.add_argument('--metric', default='f1_macro', help='Metric to plot')
    parser.add_argument('-v', '--version', type=int, default=7, help='Plot version')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
