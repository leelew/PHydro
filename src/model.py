import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K


class LSTM(Model):
    """LSTM with single task"""

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(8*cfg["n_filter_factors"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x


class MTLLSTM(Model):
    """LSTM with multi-tasks"""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(8*cfg["n_filter_factors"],
                                 return_sequences=False,
                                 name='shared_layer',
                                 recurrent_dropout=cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))

    def call(self, inputs):
        x = self.lstm(inputs)
        pred = []
        for i in range(self.num_out):
            pred.append(self.head[i](x))
        return tf.concat(pred, axis=-1)


class MTLHardLSTM(MTLLSTM):
    """LSTM with hard physical constrain through residual layer"""

    def __init__(self, cfg, resid_task_idx):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.resid_task_idx = resid_task_idx
        self.shared_layer = LSTM(8*cfg["n_filter_factors"],
                                 return_sequences=False,
                                 name='shared_layer',
                                 recurrent_dropout=cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]-1):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))

    def call(self, inputs, ):
        x = self.lstm(inputs)
        pred = []
        for i in range(self.num_out):
            pred.append(self.head[i](x))
        tf.concat(pred, axis=-1)
        # TODO: reverse normalization

        # TODO: residual layers
        inputs[:, :, 0]+inputs[:, :, 1]-tf.math.reduce_sum()
        return tf.concat(pred, axis=-1)
