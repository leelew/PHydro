from numpy import gradient
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
import tensorflow.keras.backend as K
import numpy as np



class VanillaLSTM(Model):
    """LSTM with single task"""

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(20,#8*cfg["n_filter_factors"], 
                         return_sequences=True, 
                         recurrent_dropout=cfg["dropout_rate"])
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


class MassConsLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        return super().build(input_shape)
    
    def get_config(self):
        return super().get_config()

    def call(self, inputs, aux, resid_idx):
        # 1. Concat empty tensor to inputs based on resid_idx
        empty = inputs[:,0:1]
        if resid_idx == 0: 
            inputs = tf.concat([empty, inputs], axis=-1)
        elif resid_idx == 6: 
            inputs = tf.concat([inputs, empty], axis=-1)
        else: 
            inputs = tf.concat([inputs[:,:resid_idx], empty, inputs[:,resid_idx+1:]], axis=-1)

        # 2. Tranform soil moisture in unit mm
        soil_depth = [70, 210, 720, 1864.6] # mm
        swvl1 = tf.multiply(inputs[:,0], soil_depth[0])
        swvl2 = tf.multiply(inputs[:,1], soil_depth[1])
        swvl3 = tf.multiply(inputs[:,2], soil_depth[2])
        swvl4 = tf.multiply(inputs[:,3], soil_depth[3])
        et = inputs[:,4]
        rnof = inputs[:,5]
        hydro = [swvl1, swvl2, swvl3, swvl4, et, rnof]
        inputs = tf.stack(hydro, axis=-1)

        # 3. Calculate residual outputs
        if resid_idx == 0: 
            inputs = inputs[:,1:]
        elif resid_idx == 6: 
            inputs = inputs[:,:-1]
        else: 
            inputs = tf.concat([inputs[:,:resid_idx], inputs[:,resid_idx+1:]], axis=-1)

        resid = aux[:,0:1]+aux[:,1:2]-K.sum(inputs, axis=-1, keepdims=True)

        if resid_idx == 0: 
            inputs = tf.concat([resid, inputs], axis=-1)
        elif resid_idx == 6: 
            inputs = tf.concat([inputs, resid], axis=-1)
        else: 
            inputs = tf.concat([inputs[:,:resid_idx], resid, inputs[:,resid_idx+1:]], axis=-1)

        # 4. Turn soil moisture to mm3/mm3
        soil_depth = [70, 210, 720, 1864.6] # mm
        swvl1 = tf.divide(inputs[:,0], soil_depth[0])
        swvl2 = tf.divide(inputs[:,1], soil_depth[1])
        swvl3 = tf.divide(inputs[:,2], soil_depth[2])
        swvl4 = tf.divide(inputs[:,3], soil_depth[3])
        et = inputs[:,4]
        rnof = inputs[:,5]
        hydro = [swvl1, swvl2, swvl3, swvl4, et, rnof]
        inputs = tf.stack(hydro, axis=-1)
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1] + 1)


class MTLHardLSTM(MTLLSTM):
    """LSTM with hard physical constrain through residual layer"""

    def __init__(self, cfg, resid_idx, scaler):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.resid_idx = resid_idx
        self.shared_layer = LSTM(8*cfg["n_filter_factors"],
                                 return_sequences=False,
                                 name='shared_layer',
                                 recurrent_dropout=cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]-1):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))
        self.resid_layer = MassConsLayer()

    def call(self, inputs, aux):
        x = self.lstm(inputs)
        pred = []
        for i in range(self.num_out-1):
            pred.append(self.head_layers[i](x))
        pred = tf.concat(pred, axis=-1)
        # TODO: reverse normalization

        pred = self.resid_layer(pred, aux, self.resid_idx)
        return pred

