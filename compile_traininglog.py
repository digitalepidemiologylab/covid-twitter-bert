import pandas as pd
from google.cloud import storage
import os
import json
import logging


client = storage.Client()
bucket = client.bucket('cb-tpu-projects')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def get_run_logs():
    blobs = bucket.list_blobs(prefix='covid-bert/finetune/runs/')
    run_logs = []
    for blob in blobs:
        if blob.name.endswith('run_logs.json'):
            run_log = json.loads(blob.download_as_string())
            run_logs.append(run_log)
    return run_logs

def main():
    logger.info('Collecting run logs...')
    run_logs = get_run_logs()
    logger.info(f'... loaded a total of {len(run_logs)} run logs')
    df = pd.DataFrame(run_logs)
    f_path = os.path.join('.', 'traininglog.csv')
    logger.info(f'Writing to file {f_path}')
    df.to_csv(f_path, index=False)

if __name__ == "__main__":
    main()
