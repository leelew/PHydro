import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss, mean_squared_error

from utils import make_CoLM_soil_depth


class NaNMSELoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.idx = cfg["resid_idx"]

    def call(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        return mean_squared_error(y_true, y_pred)


class MassConsLoss(Loss):
    def __init__(self, cfg, mean, std):
        self.mean = mean
        self.std = std
        self.scale = cfg["lam"]
        super().__init__()

    # NOTE: Loss class must use call rather than __call__
    def call(self, aux, pred): #(b,6), (b,2,6)
        # init
        depth, zi = make_CoLM_soil_depth() # m/cm
        soil_depth = [70, 210, 720, 10*(zi[9]-100)] # mm 
        aux = tf.cast(aux, 'float32')
        pred = tf.cast(pred, 'float32')
        std = tf.cast(self.std, 'float32')
        mean = tf.cast(self.mean, 'float32')
        pred_prev, pred_now = pred[:,0], pred[:,1]
        mask = aux == aux
        aux = aux[mask]
        pred_now = pred_now[mask]
        pred_prev = pred_prev[mask]
        std = std[mask]
        mean = mean[mask]
        
        # reverse scaling
        pred_prev = math.multiply(pred_prev, std)+mean
        pred_now = math.multiply(pred_now, std)+mean

        # cal water budget
        swvl_prev = math.multiply(pred_prev[:,:4], soil_depth) # (b,4)
        swvl_now = math.multiply(pred_now[:,:4], soil_depth) # (b,4)
        delta_swvl = math.reduce_sum(swvl_now-swvl_prev, axis=-1) #(b,)
        w_b = aux-delta_swvl-pred_now[:,-2]-pred_now[:,-1] #(b,)
        # NOTE: `y[0]+math.reduce_sum(y[1:])` != `math.reduce_sum(y)`
        return math.multiply(self.scale, math.reduce_mean(w_b))


    