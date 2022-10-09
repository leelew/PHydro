from numpy import gradient
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K


class VanillaLSTM(Model):
    """LSTM with single task"""

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(8*cfg["n_filter_factors"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


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
        x = self.lstm(inputs)  # shared layer
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head[i](x))
        return tf.concat(pred, axis=-1)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            pred = self(x)
            loss = []
            for i in range(self.num_out):
                loss.append(self.compiled_loss(y, y_pred))
        trainable_vars = self.trainable_variables
        for i in range(self.num_out):
            vars = [v for v in trainable_vars if "head_layer" +
                    str(i+1) in v.name]
            grads = tape.gradient(loss[i], vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


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

    @tf.function
    def train_step(self, data):
        pass
