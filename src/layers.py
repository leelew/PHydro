import tensorflow as tf
from tensorflow import math
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import mean_squared_error


class WeightedMultiLossLayer(Layer):
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
        super().build(input_shape)

    def call(self, y_true, y_pred):
        loss_sum = 0
        # cal seperate RMSE loss
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        self.metrics_ = []

        for i in range(self.num_out):
            a,b = y_true[:,i], y_pred[:,i]
            mask = a == a
            a, b = a[mask], b[mask]
            self.metrics_.append(math.sqrt(mean_squared_error(a,b)))

        for i, loss in enumerate(self.metrics_):
            loss_sum += 0.5 / (self.log_vars[i] ** 2) * loss + \
                tf.math.log(1 + self.log_vars[i] ** 2)
        #self.add_loss(loss_sum)
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
        depth, zi = make_CoLM_soil_depth() # m/cm
        soil_depth = [70, 210, 720, 10*(zi[9]-100)] # mm
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
        mass_resid = aux-math.reduce_sum(inputs, axis=-1)
        mass_resid = mass_resid[:, tf.newaxis]
        # Output
        inputs = self._fill_matrix(inputs, self.idx, mass_resid)
        swvl = tf.divide(inputs[:, :4], soil_depth)  # mm3/mm3
        inputs = tf.concat([swvl, inputs[:, 4:]], axis=-1)
        inputs = math.divide(inputs-mean, std)
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1] + 1)

