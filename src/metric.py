import tensorflow as tf
from tensorflow.keras.metrics import Metric


class NSEMetrics(Metric):
    def __init__(self, cfg):
        super().__init__()
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        self.model = cfg["model_name"]
        if self.model == 'single_task': 
            self.num_out = 1
        else: 
            self.num_out = cfg["num_out"]

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)
        
        self.metrics_ = []
        for i in range(self.num_out):
            a,b = y_true[:,:,i], y_pred[:,:,i]
            mask = a == a
            a, b = a[mask], b[mask]
            unexplained_error = tf.reduce_sum(tf.square(a-b))
            total_error = tf.reduce_sum(tf.square(a - tf.reduce_mean(a)))
            r2 = 1. - tf.divide(unexplained_error, total_error)
            self.metrics_.append(r2)

    def result(self):
        if self.model == 'single_task':
            return {"loss": self.metrics_[0]}
        else:
            return {"0": self.metrics_[0],
                    "1": self.metrics_[1],
                    "2": self.metrics_[2],
                    "3": self.metrics_[3],
                    "4": self.metrics_[4]}
