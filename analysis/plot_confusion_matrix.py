import os
import logging
import sklearn.metrics
import pandas as pd
from utils.analysis_helpers import find_project_root
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_train_logs():
    f_path = os.path.join(find_project_root(), 'traininglog.csv')
    return pd.read_csv(f_path)


def main():
    df = get_train_logs()

    __import__('pdb').set_trace()
    f_path = os.path.join(find_project_root(), 'output', run)
    if not os.path.isdir(f_path):
        raise FileNotFoundError(f'Could not find run directory {f_path}')
    test_output_file = os.path.join(find_project_root(), 'output', run, 'test_output.csv')
    if not os.path.isfile(test_output_file):
        raise FileNotFoundError(f'No file {test_output_file} found for run {run}. Pass the option `write_test_output: true` when training the model.')
    df = pd.read_csv(test_output_file)
    label_mapping = get_label_mapping(f_path)
    labels = list(label_mapping.keys())
    cnf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction)
    df = pd.DataFrame(cnf_matrix, columns=labels, index=labels)
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    sns.heatmap(df, ax=ax, annot=True, fmt='d', annot_kws={"fontsize":8})
    ax.set(xlabel='predicted label', ylabel='true label')
    save_fig(fig, f_path, 'confusion_matrix')


if __name__ == "__main__":
    main()
