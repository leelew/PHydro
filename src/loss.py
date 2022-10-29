import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss, mean_squared_error


class RMSELoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))


class MassConserveLoss(Loss):
    def __init__(self, cfg, mean, std):
        self.mean = mean
        self.std = std
        self.scale = cfg["lam"]
        super().__init__()

    # NOTE: Loss class must use call rather than __call__
    def call(self, aux, pred):
        # init
        soil_depth = [70, 210, 720, 1864.6]  # mm
        aux = tf.cast(aux, 'float32')
        pred = tf.cast(pred, 'float32')
        std = tf.cast(self.std, 'float32')
        mean = tf.cast(self.mean, 'float32')
        # reverse scaling
        pred = math.multiply(pred, std)+mean
        # cal mass conserve loss
        swvl1 = math.multiply(pred[:, 0], soil_depth[0])
        swvl2 = math.multiply(pred[:, 1], soil_depth[1])
        swvl3 = math.multiply(pred[:, 2], soil_depth[2])
        swvl4 = math.multiply(pred[:, 3], soil_depth[3])
        swvl = swvl1+swvl2+swvl3+swvl4
        phy_loss = tf.abs(aux[:, 0]+aux[:, 1]-swvl-pred[:, 4]-pred[:, 5])
        return math.multiply(self.scale, math.reduce_mean(phy_loss))
