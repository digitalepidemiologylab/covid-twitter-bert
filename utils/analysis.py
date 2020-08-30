import json
import os
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import logging
import glob
import pandas as pd
from collections import defaultdict
import re


logger = logging.getLogger(__name__)

def get_run_logs(pattern=None, run_type='finetune', bucket_name='my-bucket', project_name='covid-bert'):
    f_names = glob.glob(os.path.join(find_project_root(), 'data', bucket_name, project_name, run_type, '*', 'run_logs.json'))
    df = []
    for f_name in f_names:
        run_name = f_name.split('/')[-2]
        if pattern is None or re.search(pattern, run_name):
            with open(f_name, 'r') as f:
                df.append(json.load(f))
    df = pd.DataFrame(df)
    if len(df) > 0:
        df['created_at'] = pd.to_datetime(df.created_at)
        df.sort_values('created_at', inplace=True, ascending=True)
    return df

def get_summary_logs(pattern=None, dataset_type='train', bucket_name='my-bucket', project_name='covid-bert'):
    f_names = glob.glob(os.path.join(find_project_root(), 'data', bucket_name, project_name, 'pretrain', '*', 'summaries', dataset_type, '*'))
    files = []
    for f_name in f_names:
        run_name = f_name.split('/')[-4]
        if pattern is None or re.search(pattern, run_name):
            files.append(f_name)
    if len(files) == 0:
        return pd.DataFrame()
    df = pd.DataFrame()
    for f_name in files:
        run_name = f_name.split('/')[-4]
        summary_data = defaultdict(dict)
        for e in summary_iterator(f_name):
            for v in e.summary.value:
                if v.simple_value:
                    summary_data[v.tag].update({int(e.step): float(v.simple_value)})
                else:
                    summary_data[v.tag].update({int(e.step): float(tensor_util.MakeNdarray(v.tensor))})
        summary_data = pd.DataFrame(summary_data)
        summary_data['run_name'] = run_name
        df = pd.concat([df, summary_data], axis=0)
    return df

def find_project_root():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

