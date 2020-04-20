import logging
import json
import os
import csv
import argparse
import fcntl
import pandas as pd

logger = logging.getLogger(__name__)

def save_to_json(data, f_name):
    with open(f_name, mode='w') as f:
        json.dump(data, f, cls=JSONEncoder, indent=4)
    logger.info(f'Wrote log to json file {f_name}')

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
