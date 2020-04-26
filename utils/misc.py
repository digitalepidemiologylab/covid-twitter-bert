import logging
import json
import os
import csv
import argparse
import fcntl
import pandas as pd
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

def save_to_json(data, f_name):
    with tf.io.gfile.GFile(f_name, 'w') as writer:
        writer.write(json.dumps(data, cls=JSONEncoder, indent=4))

def append_to_csv(data, f_name_local, f_name_remote):
    df = pd.DataFrame.from_dict({x: [y] for x, y in data.items()})
    if os.path.isfile(f_name_local):
        df_old = pd.read_csv(f_name_local)
        df = pd.concat([df_old, df])
        f = open(f_name_local)
        fcntl.flock(f, fcntl.LOCK_EX)
        df.to_csv(f_name_local, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    else:
        df.to_csv(f_name_local, index=False)
    logger.info(f'Wrote log to csv {f_name_local}')
    # write to cloud storage
    df.to_csv(f_name_remote, index=False)
    logger.info(f'Wrote log to csv {f_name_remote}')

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class ArgParseDefault(argparse.ArgumentParser):
    """Simple wrapper which shows defaults in help"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def add_bool_arg(parser, name, default=False, help=''):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--do_not_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})

