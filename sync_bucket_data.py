import pandas as pd
from google.cloud import storage
import os
import glob
import json
import logging
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)
DATA_DIR = os.path.join('data')


def sync(run_type, bucket, args):
    blobs = bucket.list_blobs(prefix=f'{args.project_name}/{run_type}/runs/')
    dest_folder = os.path.join(DATA_DIR, args.bucket_name, args.project_name, run_type)
    count = 0
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    for blob in blobs:
        run_name = blob.name.split('/')[3]
        if 'run_logs.json' in blob.name:
            folder = os.path.join(dest_folder, run_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
                f_out = os.path.join(folder, 'run_logs.json')
                if not os.path.isfile(f_out):
                    logger.info(f'Downloading file {blob.name} to {f_out}')
                    blob.download_to_filename(f_out)
                    count += 1
        elif not args.exclude_summaries and 'events' in blob.name:
            _type = blob.name.split('/')[-2]
            if _type in ['metrics', 'train', 'eval']:
                folder = os.path.join(dest_folder, run_name, 'summaries', _type)
            else:
                folder = os.path.join(dest_folder, run_name, 'summaries')
            if not os.path.isdir(folder):
                os.makedirs(folder)
            f_out = os.path.join(folder, os.path.basename(blob.name))
            if not os.path.isfile(f_out):
                logger.info(f'Downloading file {blob.name} to {f_out}')
                blob.download_to_filename(f_out)
                count += 1
    if count > 0:
        logger.info(f'Collected a total of {count:,} files of type {run_type}')
    else:
        logger.info(f'All files of type {run_type} are up-to-date!')

def main(args):
    client = storage.Client()
    bucket = client.bucket(args.bucket_name)

    if 'finetune' in args.types:
        logger.info(f'Collecting finetune run logs...')
        sync('finetune', bucket, args)
    if 'pretrain' in args.types:
        logger.info(f'Collecting pretrain run logs...')
        sync('pretrain', bucket, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', required=True, help='Bucket name')
    parser.add_argument('--project_name', default='covid-bert', help='Name of subfolder in Google bucket')
    parser.add_argument('--types', default=['pretrain', 'finetune'], choices=['pretrain', 'finetune'], help='Types of training logs')
    parser.add_argument('--exclude_summaries', action='store_true', help='Exclude summaries')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
