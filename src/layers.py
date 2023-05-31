import tensorflow as tf
from tensorflow import math
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import mean_squared_error
from utils import make_CoLM_soil_depth


class AdapMultiLossLayer(Layer):
    """Cal uncertainty weighted multi-task loss"""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
    
    def build(self, input_shape=None):
        self.log_vars = []
        for i in range(self.num_out):
            self.log_vars += [self.add_weight(name='log_var'+str(i), 
                                              shape=(1,), 
                                              initializer=Constant(1.),
                                              trainable=True,
                                              dtype='float64')]
        print(self.log_vars)
        super().build(input_shape)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        # cal MSE
        self.metrics_ = []
        for i in range(self.num_out):
            a,b = y_true[:,:,i], y_pred[:,:,i]
            mask = a == a
            a, b = a[mask], b[mask]
            mask = b == b
            a, b = a[mask], b[mask]
            self.metrics_.append(mean_squared_error(a,b))
        
        # cal uncertainty weighted loss
        # 
        # Reference:
        #   Kendall et al. 2019. Multi-Task Learning Using Uncertainty to 
        #   Weigh Losses for Scene Geometry and Semantics. CVPR.
        loss_sum = 0
        for i, loss in enumerate(self.metrics_):
            loss_sum += 0.5 / (self.log_vars[i] ** 2) * loss + \
                tf.math.log(1 + self.log_vars[i] ** 2)
            #loss_sum+=tf.math.exp(-self.log_vars[i])*loss+self.log_vars
        return loss_sum


class MassConsLayer(Layer):
    """Mass conserve layer"""

    def __init__(self, cfg):
        super().__init__()
        self.idx = cfg["resid_idx"]
        self.num_out = cfg["num_out"]

    def _fill_matrix(self, x, idx, res=None):
        if res is not None:
            empty = res
        else:
            empty = x[:,0:1]
        if idx == 0:
            x = tf.concat([empty, x], axis=-1)
        elif idx == self.num_out-1:
            x = tf.concat([x, empty], axis=-1)
        else:
            x = tf.concat([x[:,:idx], empty, x[:,idx:]], axis=-1)
        return x

    def _slice_matrix(self, x, idx):
        if idx == 0:
            x = x[:, 1:]
        elif idx == self.num_out:
            x = x[:, :-1]
        else:
            x = tf.concat([x[:, :idx], x[:, idx+1:]], axis=-1)
        return x

    def call(self, pred_prev, pred_now, aux, mean, std):
        # init
        pred_prev = tf.cast(pred_prev,'float32') #(b,6)
        pred_prev_bk = pred_prev
        pred_now = tf.cast(pred_now,'float32') #(b,5)
        aux = tf.cast(aux,'float32') # (b,)
        mean = tf.cast(mean,'float32') # (b,1,5)
        std = tf.cast(std,'float32') #(b,1,5)
        soil_depth = [70, 210, 720] # mm
        
        # Concat empty tensor to inputs based on idx
        pred_now = self._fill_matrix(pred_now, self.idx) # (b,6)
        # reverse normalized forecasts
        pred_now = math.multiply(pred_now, std) + mean #(b,6)
        pred_prev = math.multiply(pred_prev, std) + mean #(b,6)
        # Transform soil moisture in unit mm
        swvl_prev = math.multiply(pred_prev[:,:3], soil_depth) #(b,4)
        swvl_now = math.multiply(pred_now[:,:3], soil_depth) #(b,4)
        wat_prev = aux+math.reduce_sum(swvl_prev, axis=-1) #(b,)
        pred_now = tf.concat([swvl_now, pred_now[:, 3:]], axis=-1) #(b,6)
        # 
        pred_now = self._slice_matrix(pred_now, self.idx) #(b,5)
        mass_resid = wat_prev-math.reduce_sum(pred_now, axis=-1) #(b,)
        mass_resid = mass_resid[:, tf.newaxis] #(b,1)
        # Output
        pred_now = self._fill_matrix(pred_now, self.idx, mass_resid) #(b,6)
        swvl = tf.divide(pred_now[:, :3], soil_depth)  # mm3/mm3
        pred_now = tf.concat([swvl, pred_now[:, 3:]], axis=-1) #(b,6)
        pred_now = math.divide(pred_now-mean, std) #(b,6)
        pred = tf.stack([pred_prev_bk, pred_now], axis=1) #(b,2,6)
        return pred


