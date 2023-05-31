import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss, mean_squared_error

from utils import make_CoLM_soil_depth


class WeightedMSELoss(Loss):
    """weighted multi-task MSE loss"""

    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.num_out = cfg["num_out"]
        self.weight = [cfg["scaling_factor"]]*cfg["num_out"]
        if cfg["model_name"] not in ["single_task","multi_tasks_v2"]:
            self.weight[cfg["main_idx"]] = 1
        self.resid_idx = cfg["resid_idx"]

    def call(self, y_true, y_pred):
        if self.model_name == 'single_task':
            mask = y_true == y_true
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            return mean_squared_error(y_true, y_pred)
        elif self.model_name == 'hard_multi_tasks_v2':
            self.metrics_ = []
            for i in range(self.num_out):
                if i == self.resid_idx:
                    #a,b = y_true[:,0,i], y_pred[:,0,i] # offline
                    a,b = y_true[:,:,i], y_pred[:,:,i] # online
                else:
                    a,b = y_true[:,:,i], y_pred[:,:,i]
                mask = a == a
                a, b = a[mask], b[mask]
                mask = b == b
                a, b = a[mask], b[mask]
                self.metrics_.append(mean_squared_error(a,b))
            loss_sum = 0
            for i, loss in enumerate(self.metrics_):
                loss_sum += self.weight[i]*loss
            return loss_sum 
        else:
            self.metrics_ = []
            for i in range(self.num_out):
                a,b = y_true[:,:,i], y_pred[:,:,i]
                mask = a == a
                a, b = a[mask], b[mask]
                mask = b == b
                a, b = a[mask], b[mask]
                self.metrics_.append(mean_squared_error(a,b))

            loss_sum = 0
            for i, loss in enumerate(self.metrics_):
                loss_sum += self.weight[i]*loss
            return loss_sum


class ValMSELoss(Loss):
    """weighted multi-task MSE loss"""

    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.num_out = cfg["num_out"]

    def call(self, y_true, y_pred):
        if self.model_name == 'single_task':
            mask = y_true == y_true
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            return mean_squared_error(y_true, y_pred)
        
        self.metrics_ = []
        for i in range(self.num_out):
            a,b = y_true[:,:,i], y_pred[:,:,i]
            mask = a == a
            a, b = a[mask], b[mask]
            mask = b == b
            a, b = a[mask], b[mask]
            self.metrics_.append(mean_squared_error(a,b))

        loss_sum = 0
        for i, loss in enumerate(self.metrics_):
            loss_sum+=loss
        return loss_sum


class ValMainMSELoss(Loss):
    """weighted multi-task MSE loss"""

    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.main_idx = cfg["main_idx"]

    def call(self, y_true, y_pred):
        if self.model_name == 'single_task':
            mask = y_true == y_true
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            return mean_squared_error(y_true, y_pred)
        
        a,b = y_true[:,:,self.main_idx], y_pred[:,:,self.main_idx]
        mask = a == a
        a, b = a[mask], b[mask]
        mask = b == b
        a, b = a[mask], b[mask]
        return mean_squared_error(a,b)


class MassConsLoss(Loss):
    """Physical loss for soft constrain."""

    def __init__(self, cfg, mean, std):
        self.mean = mean
        self.std = std
        self.scale = cfg["lam"]
        super().__init__()

    # NOTE: Loss class must use call rather than __call__
    def call(self, aux, pred): #(b,6), (b,2,6)
        # init
        soil_depth = [70, 210, 720] # mm 
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
        swvl_prev = math.multiply(pred_prev[:,:3], soil_depth) # (b,4)
        swvl_now = math.multiply(pred_now[:,:3], soil_depth) # (b,4)
        delta_swvl = math.reduce_sum(swvl_now-swvl_prev, axis=-1) #(b,)
        w_b = aux-delta_swvl-pred_now[:,-2]-pred_now[:,-1] #(b,)
 
        # NOTE: `y[0]+math.reduce_sum(y[1:])` != `math.reduce_sum(y)`
        return math.multiply(self.scale, math.reduce_mean(math.abs(w_b)))


    