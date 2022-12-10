"""
All model structure

<<<<<<<< HEAD
Author: Lu Li 
11/1/2022 - V1.0
12/9/2022 - V2.0
"""

import tensorflow as tf
from tensorflow import math
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from utils import make_CoLM_soil_depth
from layers import WeightedMultiLossLayer, MassConsLayer



class STModel(Model):
    """single task model"""

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(cfg["hidden_size"], return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.drop(x)
        # we only predict the last two steps
        x = self.dense(x[:,-2:]) 
        return x


class MTLModel_v1(Model):
    """multitasks model with average loss"""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"], return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))     

    def call(self, inputs):
        x = self.shared_layer(inputs)  # shared layer
        x = self.drop(x)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1)
        return pred


class MTLModel_v2(Model):
    """multitasks model with adaptive loss"""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"], return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1))) 
        self.loss_layer = WeightedMultiLossLayer(cfg) 

    def call(self, inputs, y_true=None):
        x = self.shared_layer(inputs)  # shared layer
        x = self.drop(x)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1)

        if y_true is not None: # train mode
            #FIXME: Only cal loss on last step
            loss_sum = self.loss_layer(y_true, pred)
            return pred, loss_sum
        else: # inference mode
            return pred


class HardMTLModel_v1(Model):
    """multitasks model with hard physical constrain through redistribute layer."""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"],return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers = []
        for i in range(cfg["num_out"]):
            self.head_layers.append(Dense(1, name='head_layer_'+str(i+1)))

    def call(self, inputs, aux, mean, std):
        x = self.shared_layer(inputs)  # shared layer
        x = self.drop(x)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1) #(b, 2, 6)

        # -------------------------
        # redistribute water budget
        # -------------------------
        depth, zi = make_CoLM_soil_depth() # m/cm
        soil_depth = [70, 210, 720, 10*(zi[9]-100)] # mm 
        pred_prev_save, pred_now = pred[:,0], pred[:,1]  #(b,6)
        pred_prev = math.multiply(pred_prev_save, std) + mean
        pred_now = math.multiply(pred_now, std) + mean
        print(tf.shape(pred_prev))

        # cal water budget
        swvl_prev = math.multiply(pred_prev[:,:4], soil_depth) # (b,4)
        swvl_now = math.multiply(pred_now[:,:4], soil_depth) # (b,4)
        delta_swvl = math.reduce_sum(swvl_now-swvl_prev, axis=-1) #(b,)
        w_b = aux-delta_swvl-pred_now[:,-2]-pred_now[:,-1] #(b,)

        # cal ratio and distribute
        pred_new = []
        w_a = math.reduce_sum(swvl_now, axis=-1) #(b)
        for i in range(4):
            ratio = math.divide(swvl_now[:,i], w_a)
            water_add = math.multiply(w_b, ratio)
            pred_new.append((water_add+swvl_now[:,i])/soil_depth[i])
        pred_new.append(pred_now[:,-2])
        pred_new.append(pred_now[:,-1])
        pred_new = tf.stack(pred_new, axis=-1) #(b,6)
        pred_new = math.divide(pred_new-mean, std)
        pred = tf.stack([pred_prev_save, pred_new], axis=1) #(b,2,6)
        print(tf.shape(pred))
        return pred #(b,2,6)


class MTLHardLSTM_v2(Model):
    """LSTM with hard physical constrain through residual layer."""

    def __init__(self, cfg):
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

    def call(self, inputs, aux, mean, std):
        x = self.shared_layer(inputs)
        x = self.drop(x)
        pred = []
        for i in range(self.num_out-1):
            pred.append(self.head_layers[i](x))
        pred = tf.concat(pred, axis=-1)
        pred = self.resid_layer(pred, aux, mean, std)
        return pred




