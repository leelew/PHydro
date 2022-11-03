import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss, mean_squared_error
from tensorflow.keras.metrics import Metric


class RMSELoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.sub = list(range(cfg["num_out"]))
        self.sub.pop(cfg["resid_idx"])
        self.idx = cfg["resid_idx"]

    def call(self, y_true, y_pred):
        if self.model_name == 'hard_multi_tasks_v3':
            y_true = tf.gather(y_true, self.sub, axis=-1)
            y_pred = tf.gather(y_pred, self.sub, axis=-1)
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        return math.sqrt(mean_squared_error(y_true, y_pred))


class MassConsLoss(Loss):
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
        swvl = math.multiply(pred[:, :4], soil_depth)
        # unnormalized all mm/day
        pred = tf.concat([swvl, pred[:, 4:]], axis=-1)
        # NOTE: `y[0]+math.reduce_sum(y[1:])` != `math.reduce_sum(y)`
        phy_loss = tf.abs(aux-pred[:,0]-math.reduce_sum(pred[:,1:], axis=-1))
        return math.multiply(self.scale, math.reduce_mean(phy_loss))


class RMetrics(Metric):
    def __init__(self, cfg):
        super().__init__()
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        self.model = cfg["model_name"]

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        self.metrics_ = []
        if self.model == 'single_task':
            num_out = 1
        else:
            num_out = 6

        for i in range(num_out):
            a,b = y_true[:,i], y_pred[:,i]
            mask = a == a
            a, b = a[mask], b[mask]
            unexplained_error = tf.reduce_sum(tf.square(a-b))
            total_error = tf.reduce_sum(tf.square(a - tf.reduce_mean(a)))
            r2 = 1. - tf.divide(unexplained_error, total_error)
            self.metrics_.append(r2)

    def result(self):
        if self.model == 'single_task':
            return {"all": self.metrics_[0]}
        else:
            return {"SWVL_1": self.metrics_[0],
                    "SWVL_2": self.metrics_[1],
                    "SWVL_3": self.metrics_[2],
                    "SWVL_4": self.metrics_[3],
                    "ET": self.metrics_[4],
                    "R": self.metrics_[5]}
