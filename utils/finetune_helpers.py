import numpy as np
import sklearn.metrics
import os
import logging
import time
import tensorflow as tf


logger = logging.getLogger(__name__)

class Metrics(tf.keras.callbacks.Callback):
    """Metrics to compute as a callback"""

    def __init__(self, eval_data_fn, label_mapping):
        super().__init__()
        self.eval_data = eval_data_fn()
        self.label_mapping = label_mapping
        self.scores = []
        self.predictions = []

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        y_true = np.concatenate([label.numpy() for _, label in self.eval_data])
        preds = self.model.predict(self.eval_data)
        y_pred = tf.argmax(preds, axis=1).numpy()
        scores = self.performance_metrics(y_true, y_pred, label_mapping=self.label_mapping)
        logger.info(f'Scores after epoch {epoch}:\n{scores}')
        # store scores and predictions for later
        self.scores.append(scores)
        self.predictions.append(list(y_pred))

    def performance_metrics(self, y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
        """
        Compute performance metrics
        """
        def _compute_performance_metric(scoring_function, m, y_true, y_pred):
            # compute averaging
            for av in averaging:
                if av is None:
                    metrics_by_class = scoring_function(y_true, y_pred, average=av, labels=labels)
                    for i, class_metric in enumerate(metrics_by_class):
                        if label_mapping is None:
                            label_name = labels[i]
                        else:
                            label_name = label_mapping[labels[i]]
                        scores['scores_by_label'][m + '_' + str(label_name)] = class_metric
                else:
                    scores[m + '_' + av] = scoring_function(y_true, y_pred, average=av, labels=labels)

        if averaging is None:
            averaging = ['micro', 'macro', 'weighted', None]
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'cohen_kappa']
        scores = {'scores_by_label': {}}
        if label_mapping is None:
            # infer labels from data
            labels = sorted(list(set(y_true + y_pred)))
        else:
            labels = sorted(list(label_mapping.keys()))
        if len(labels) <= 2:
            # binary classification
            averaging += ['binary']
        for m in metrics:
            if m == 'accuracy':
                scores[m] = sklearn.metrics.accuracy_score(y_true, y_pred)
            elif m == 'precision':
                _compute_performance_metric(sklearn.metrics.precision_score, m, y_true, y_pred)
            elif m == 'recall':
                _compute_performance_metric(sklearn.metrics.recall_score, m, y_true, y_pred)
            elif m == 'f1':
                _compute_performance_metric(sklearn.metrics.f1_score, m, y_true, y_pred)
            elif m == 'matthews_corrcoef':
                scores[m] = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
            elif m == 'cohen_kappa':
                scores[m] = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
        return scores
