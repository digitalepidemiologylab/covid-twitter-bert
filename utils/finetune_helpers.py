import numpy as np
import sklearn.metrics
import os
import logging
import time
import tensorflow as tf


logger = logging.getLogger(__name__)

class Metrics(tf.keras.callbacks.Callback):
    """Metrics to compute as a callback"""

    def __init__(self, eval_data_fn, label_mapping, logdir, eval_steps, eval_batch_size, validation_freq):
        super().__init__()
        self.eval_data = eval_data_fn()
        self.label_mapping = label_mapping
        self.scores = []
        self.predictions = []
        self.logdir = logdir
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.validation_freq = validation_freq

    def on_epoch_end(self, epoch, logs):
        if isinstance(self.validation_freq, int):
            if (epoch + 1) % self.validation_freq != 0:
                logger.info(f'Not running eval for epoch {epoch+1}')
                return
        elif isinstance(self.validation_freq, list):
            if epoch + 1 not in self.validation_freq:
                logger.info(f'Not running eval for epoch {epoch+1}')
                return
        else:
            raise ValueError('Incorrect type for validation frequency')
        logger.info('Computing metrics on validation set...')
        t_s = time.time()
        y_true = np.concatenate([label.numpy() for _, label in self.eval_data])
        preds = self.model.predict(self.eval_data, steps=self.eval_steps, batch_size=self.eval_batch_size)
        y_pred = tf.argmax(preds, axis=1).numpy()
        scores = self.performance_metrics(y_true, y_pred, label_mapping=self.label_mapping)
        # add to summary writer
        metrics_writer = tf.summary.create_file_writer(self.logdir)
        metrics_writer.set_as_default()
        for metric, value in scores.items():
            if metric != 'scores_by_label':
                tf.summary.scalar(metric, data=value, step=epoch)
        # store scores and predictions for later
        self.scores.append(scores)
        self.predictions.append(list(y_pred))
        t_e = time.time()
        logger.info(f'... finished computing metrics in {int(t_e-t_s):,} s.')
        logger.info(f'Scores after epoch {epoch+1}:\n{scores}')

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
