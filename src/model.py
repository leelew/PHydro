from ast import Mod
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
    def __init__(self, cfg, resid_idx):
        super().__init__()
        self.resid_idx = resid_idx
        self.num_out = cfg["num_out"]

    def call(self, inputs, aux, mean, std):
        """
        Args
        ----
            inputs: Directly outputs of multi-task models. Notably,
                    it's z-score normalized value, and if we want
                    to predict N vars, it only contains (N-1) vars.
        """
        inputs = tf.cast(inputs, 'float32')
        aux = tf.cast(aux, 'float32')
        mean = tf.cast(mean, 'float32')
        std = tf.cast(std, 'float32')
        print(aux)

        # 1. Concat empty tensor to inputs based on resid_idx (batch, nout)
        empty = inputs[:, 0:1]
        if self.resid_idx == 0:
            inputs = tf.concat([empty, inputs], axis=-1)
        elif self.resid_idx == self.num_out:
            inputs = tf.concat([inputs, empty], axis=-1)
        else:
            inputs = tf.concat(
                [inputs[:, :self.resid_idx], empty, inputs[:, self.resid_idx+1:]], axis=-1)

        # 2. reverse normalized forecasts 
        inputs = math.multiply(inputs, std) + mean

        # 3. Transform soil moisture in unit mm
        soil_depth = [70, 210, 720, 1864.6]  # mm
        swvl1 = tf.multiply(inputs[:, 0], soil_depth[0])
        swvl2 = tf.multiply(inputs[:, 1], soil_depth[1])
        swvl3 = tf.multiply(inputs[:, 2], soil_depth[2])
        swvl4 = tf.multiply(inputs[:, 3], soil_depth[3])
        inputs = [swvl1, swvl2, swvl3, swvl4, inputs[:,4], inputs[:,5]]
        inputs = tf.stack(inputs, axis=-1)

        # 4. Calculate residual outputs
        if self.resid_idx == 0:
            inputs = inputs[:, 1:]
        elif self.resid_idx == self.num_out:
            inputs = inputs[:, :-1]
        else:
            inputs = tf.concat(
                [inputs[:, :self.resid_idx], inputs[:, self.resid_idx+1:]], axis=-1)
        mass_prev = math.reduce_sum(aux, axis=-1)
        mass_now = math.reduce_sum(inputs, axis=-1)  
        print(mass_prev)
        print(mass_now)     
        mass_resid = mass_prev-mass_now
        mass_resid = (mass_resid-mean[:,self.resid_idx])/std[:,self.resid_idx]
        mass_resid = mass_resid[:, tf.newaxis]
        if self.resid_idx == 0:
            inputs = tf.concat([mass_resid, inputs], axis=-1)
        elif self.resid_idx == self.num_out:
            inputs = tf.concat([inputs, mass_resid], axis=-1)
        else:
            inputs = tf.concat(
                [inputs[:, :self.resid_idx], mass_resid, inputs[:, self.resid_idx+1:]], axis=-1)

        # 5. Turn soil moisture to mm3/mm3
        swvl1 = tf.divide(inputs[:, 0], soil_depth[0])
        swvl2 = tf.divide(inputs[:, 1], soil_depth[1])
        swvl3 = tf.divide(inputs[:, 2], soil_depth[2])
        swvl4 = tf.divide(inputs[:, 3], soil_depth[3])
        inputs = [swvl1, swvl2, swvl3, swvl4, inputs[:,4], inputs[:,5]]
        inputs = tf.stack(inputs, axis=-1)
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
        self.resid_layer = MassConsLayer(cfg, resid_idx)
        self.soil_depth = [70, 210, 720, 1864.6] # mm


    def call(self, inputs, aux, mean, std):
        x = self.shared_layer(inputs)
        x = self.drop(x)
        pred = []
        for i in range(self.num_out-1):
            pred.append(self.head_layers[i](x))
        pred = tf.concat(pred, axis=-1)
        pred = self.resid_layer(pred, aux, mean, std)


        # test mc loss
        # reverse scaling
        pred = math.multiply(pred, std)+mean

        # cal mass conserve loss
        swvl1 = math.multiply(pred[:,0],self.soil_depth[0])
        swvl2 = math.multiply(pred[:,1],self.soil_depth[1])
        swvl3 = math.multiply(pred[:,2],self.soil_depth[2])
        swvl4 = math.multiply(pred[:,3],self.soil_depth[3])
        swvl = swvl1+swvl2+swvl3+swvl4
        phy_loss = tf.abs(aux[:,0]+aux[:,1]-swvl-pred[:,4]-pred[:,5])
        return pred, math.multiply(0.01, math.reduce_mean(phy_loss))
