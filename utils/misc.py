import logging
import json
import os
import csv
import argparse
import fcntl

logger = logging.getLogger(__name__)

def save_to_json(data, f_name):
    with open(f_name, mode='w') as f:
        json.dump(data, f, cls=JSONEncoder, indent=4)
    logger.info(f'Wrote log to json file {f_name}')

def append_to_csv(data, f_name):
    datafields = sorted(data.keys())
    def _get_dict_writer(f):
        return csv.DictWriter(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
    if not os.path.isfile(f_name):
        with open(f_name, mode='w') as f:
            # activate file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            output_writer = _get_dict_writer(f)
            output_writer.writeheader()
            # release file lock
            fcntl.flock(f, fcntl.LOCK_UN)
    with open(f_name, mode='a+') as f:
        # activate file lock
        fcntl.flock(f, fcntl.LOCK_EX)
        output_writer = _get_dict_writer(f)
        output_writer.writerow(data)
        # release file lock
        fcntl.flock(f, fcntl.LOCK_UN)
    logger.info(f'Wrote log to csv {f_name}')

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
