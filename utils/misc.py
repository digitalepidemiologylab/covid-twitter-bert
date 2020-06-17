import logging
import json
import os
import argparse
import tensorflow as tf
import numpy as np
from subprocess import PIPE, run
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

def out(command):
    """Run shell command"""
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result

def create_tpu(tpu_name, zone, tpu_type='v2-8', version='2.1'):
    # Allocate a preemptible TPU
    logger.info(f"Creating TPU of type {tpu_type} in zone {zone} called \'{tpu_name}\'")
    result = out(f"gcloud compute tpus create {tpu_name} --preemptible --zone={zone} --accelerator-type={tpu_type} --version={version}")
    if result.returncode == 1:
        logger.info(f"Failed creating the TPU with the message: {result}")
        return False
    else:
        logger.info(f"Successfully created a new TPU called \'{tpu_name}\'")
    # List ip address for TPU name
    result = out(f"gcloud compute tpus describe {tpu_name} --zone={zone} --format='get(ipAddress)'")
    if result.returncode == 1:
        logger.info(f"Something went wrong when looking up the IP of the TPU. Following error was returned: {result.stderr}")
        return False
    else:
        tpu_ip = result.stdout.strip()
        logger.info(f"TPU ipAddress is {tpu_ip}")
    
    return tpu_ip

def destroy_tpu(tpu_name, zone):
    # Destroy the TPU
    logger.info(f"Attempting to destroy TPU named {tpu_name}")
    result = out(f"gcloud compute tpus delete {tpu_name} --zone={zone} --quiet")
    if result.returncode == 1:
        logger.info(f"Something went wrong when trying to destroy the TPU. Following error was returned: {result.stderr}")
        return False
    else:
        logger.info(f"The TPU called \'{tpu_name}\' is now destroyed")

@contextmanager
def file_lock(fd):
    """ Locks FD before entering the context, always releasing the lock. """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
