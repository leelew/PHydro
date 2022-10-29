import xdrlib
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
import tensorflow.keras.backend as K
from tensorflow import math


class VanillaLSTM(Model):

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(cfg["hidden_size"], return_sequences=False)
        self.drop = Dropout(cfg["dropout_rate"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.drop(x)
        x = self.dense(x)
        return x


class MTLLSTM(Model):
    """LSTM with multi-tasks"""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"],
                                 return_sequences=False,
                                 name='shared_layer')
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))

    def call(self, inputs):
        x = self.shared_layer(inputs)  # shared layer
        x = self.drop(x)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x))
        return tf.concat(pred, axis=-1)


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
            empty = x[:, 0:1]
        if idx == 0:
            x = tf.concat([empty, x], axis=-1)
        elif idx == self.num_out:
            x = tf.concat([x, empty], axis=-1)
        else:
            x = tf.concat([x[:, :idx], empty, x[:, idx+1:]], axis=-1)
        return x

    def _slice_matrix(self, x, idx):
        if idx == 0:
            x = x[:, 1:]
        elif idx == self.num_out:
            x = x[:, :-1]
        else:
            x = tf.concat([x[:, :idx], x[:, idx+1:]], axis=-1)
        return x

    def call(self, inputs, aux, mean, std):
        """
        Args
        ----
            inputs: Directly outputs of multi-task models. Notably,
                    it's z-score normalized value, and if we want
                    to predict N vars, it only contains (N-1) vars.
        """
        # init
        inputs = tf.cast(inputs, 'float32')
        aux = tf.cast(aux, 'float32')
        mean = tf.cast(mean, 'float32')
        std = tf.cast(std, 'float32')
        soil_depth = [70, 210, 720, 1864.6]  # mm
        # Concat empty tensor to inputs based on idx (batch, nout)
        inputs = self._fill_matrix(inputs, self.idx)
        # reverse normalized forecasts
        inputs = math.multiply(inputs, std) + mean
        # Transform soil moisture in unit mm
        swvl = math.multiply(inputs[:, :4], soil_depth)
        # unnormalized all mm/day
        inputs = tf.concat([swvl, inputs[:, 4:]], axis=-1)
        # Calculate residual outputs
        inputs = self._slice_matrix(inputs, self.idx)
        mass_prev = math.reduce_sum(aux, axis=-1)
        mass_now = math.reduce_sum(inputs, axis=-1)
        mass_resid = mass_prev-mass_now
        mass_resid = mass_resid[:, tf.newaxis]
        # Output
        inputs = self._fill_matrix(inputs, self.idx, mass_resid)
        swvl = tf.divide(inputs[:, :4], soil_depth)  # mm3/mm3
        inputs = tf.concat([swvl, inputs[:, 4:]], axis=-1)
        inputs = (inputs-mean)/std
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1] + 1)


class MTLHardLSTM(Model):
    """LSTM with hard physical constrain through residual layer"""

    def __init__(self, cfg, resid_idx):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"],
                                 return_sequences=False,
                                 name='shared_layer')
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]-1):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))
        self.resid_layer = MassConsLayer(cfg)
        self.soil_depth = [70, 210, 720, 1864.6]  # mm

    def call(self, inputs, aux, mean, std):
        x = self.shared_layer(inputs)
        x = self.drop(x)
        pred = []
        for i in range(self.num_out-1):
            pred.append(self.head_layers[i](x))
        pred = tf.concat(pred, axis=-1)
        pred = self.resid_layer(pred, aux, mean, std)
        return pred
