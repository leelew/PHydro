import tensorflow as tf
from tensorflow.keras.losses import Loss
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class MSELoss(Loss):
    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float64')
        y_pred = tf.cast(y_pred, 'float64')        
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=0)
        if self.mask is None:
            return tf.math.reduce_mean(mse)
        else:
            return tf.math.reduce_mean(tf.math.multiply(mse, self.mask))


class NSELoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float64')
        y_pred = tf.cast(y_pred, 'float64')   
        #FIXME:
        return tfa.metrics.r_square.RSquare()(y_true, y_pred)



class NNSELoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float64')
        y_pred = tf.cast(y_pred, 'float64')   

        unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
        total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)))
        r2 = 1. - tf.divide(unexplained_error, total_error)
        nnse = 1- tf.divide(1, (2-r2))
        print(r2, nnse)
        return nnse




