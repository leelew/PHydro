from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class PHydroLoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.alpha = cfg["alpha"]  # weight for physical loss (0-1)
        self.num_out = cfg["num_out"]

    def call(self, y_true, y_pred, aux=None, resid_idx=None, optim_all=True): # must be call rather than __call__
        """aux: p(t), sm(t-1)"""
        if self.model_name in ["single_task", "multi-tasks"]:
            return mean_squared_error(y_true, y_pred)

        elif self.model_name in ["soft_multi_tasks"]:
            # Cal physical loss
            soil_depth = [70, 210, 720, 1864.6] # mm
            swvl1 = tf.multiply(y_pred[:,0], soil_depth[0])
            swvl2 = tf.multiply(y_pred[:,1], soil_depth[1])
            swvl3 = tf.multiply(y_pred[:,2], soil_depth[2])
            swvl4 = tf.multiply(y_pred[:,3], soil_depth[3])
            et = y_pred[:,4]
            rnof = y_pred[:,5]
            swvl = swvl1+swvl2+swvl3+swvl4
            precip = aux[:,0]
            swvl_prev = aux[:,1]
            phy_loss = precip-(swvl-swvl_prev+et+rnof)
            return tf.multiply((1-self.alpha),mean_squared_error(y_true, y_pred))+\
                tf.multiply(self.alpha, phy_loss)

        elif self.model_name == "hard_multi-tasks":
            # 1. use all outputs to optimize (UCNet in Beucler et al. 2021)
            if optim_all:
                return mean_squared_error(y_true, y_pred)
            # 2. use direct outputs (remove residual outputs) to optimize
            else:
                idx = np.arange(self.num_out)
                idx = np.delete(idx, [resid_idx])
                return mean_squared_error(y_true[:,idx], y_pred[:,idx])
