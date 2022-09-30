import tensorflow as tf
from tensorflow.keras.metrics import Metric


class RMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        m1 = tf.math.reduce_mean(y_true, axis=1, keepdims=True)
        m2 = tf.math.reduce_mean(y_pred, axis=1, keepdims=True)
        self.corr = tf.math.reduce_sum(tf.math.multiply((y_true-m1), (y_pred-m2)), axis=1)/ \
            tf.math.reduce_sum(tf.math.square((y_true-m1)), axis=1)
        self.corr_ = tf.math.reduce_mean(self.corr)

    def result(self):
        return self.corr_


class MTLRSquareMetrics(Metric):
    def __init__(self):
        super().__init__()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        self.metrics_ = []
        for i in range(6):
            unexplained_error = tf.reduce_sum(tf.square(y_true[:,i] - y_pred[:,i]))
            total_error = tf.reduce_sum(tf.square(y_true[:,i] - tf.reduce_mean(y_true[:,i], axis=0)))
            r2 = 1. - tf.divide(unexplained_error, total_error)
            self.metrics_.append(r2)
        
    def result(self):
        return {"loss_1": self.metrics_[0], 
                "loss_2": self.metrics_[1], 
                "loss_3": self.metrics_[2], 
                "loss_4": self.metrics_[3], 
                "loss_5": self.metrics_[4],
                "loss_6": self.metrics_[5]}