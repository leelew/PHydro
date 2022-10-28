import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, mean_squared_error
from tensorflow import math



class RMSELoss(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))
        # 1. use all outputs to optimize (UCNet in Beucler et al. 2021)
        if optim_all:
            return mean_squared_error(y_true, y_pred)
        # 2. use direct outputs (remove residual outputs) to optimize
        else:
            idx = np.arange(self.num_out)
            idx = np.delete(idx, [resid_idx])
            return mean_squared_error(y_true[:, idx], y_pred[:, idx])



class MassConserveLoss(Loss):
    def __init__(self, mean, std):
        self.soil_depth = [70, 210, 720, 1864.6] # mm
        self.mean = mean
        self.std = std
        super().__init__()
    
    # NOTE: Loss class must use call rather than __call__
    def call(self, aux, y_pred):
        aux = tf.cast(aux, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
        std = tf.cast(self.std, 'float32')
        mean = tf.cast(self.mean, 'float32')

        # reverse scaling
        y_pred = math.multiply(y_pred, std)+mean

        # cal mass conserve loss
        swvl1 = math.multiply(y_pred[:,0],self.soil_depth[0])
        swvl2 = math.multiply(y_pred[:,1],self.soil_depth[1])
        swvl3 = math.multiply(y_pred[:,2],self.soil_depth[2])
        swvl4 = math.multiply(y_pred[:,3],self.soil_depth[3])
        swvl = swvl1+swvl2+swvl3+swvl4
        phy_loss = tf.abs(aux[:,0]+aux[:,1]-swvl-y_pred[:,4]-y_pred[:,5])
        return math.multiply(0.01, math.reduce_mean(phy_loss))


