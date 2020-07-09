import logging
import json
import os
import argparse
import tensorflow as tf
import numpy as np
from contextlib import contextmanager
import fcntl


logger = logging.getLogger(__name__)

def save_to_json(data, f_name):
    with tf.io.gfile.GFile(f_name, 'w') as writer:
        writer.write(json.dumps(data, cls=JSONEncoder, indent=4))

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

@contextmanager
def file_lock(fd):
    """ Locks FD before entering the context, always releasing the lock. """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
