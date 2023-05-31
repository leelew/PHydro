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

from layers import MassConsLayer, AdapMultiLossLayer
from utils import make_CoLM_soil_depth


class STModel(Model):
    """single task model"""

    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(cfg["hidden_size"], return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.drop(x, training=True)
        x = self.dense(x[:,-2:])
        return x


class MTLModel_v1(Model):
    """multitasks model with manually loss"""

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
        x = self.drop(x, training=True) # use for MCD
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
        self.loss_layer = AdapMultiLossLayer(cfg) 

    def call(self, inputs, y_true=None):
        x = self.shared_layer(inputs)  # shared layer
        x = self.drop(x, training=True)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1)

        if y_true is not None: # train mode
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
        x = self.drop(x, training=True)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1) #(b, 2, 6)

        # -------------------------
        # redistribute water budget
        # -------------------------
        aux = tf.cast(aux, 'float32')
        pred = tf.cast(pred, 'float32')
        std = tf.cast(std, 'float32')
        mean = tf.cast(mean, 'float32')
        soil_depth = [70, 210, 720] # mm 
        pred_prev_save, pred_now = pred[:,0], pred[:,1]  #(b,6)

        pred_prev = math.multiply(pred_prev_save, std) + mean
        pred_now = math.multiply(pred_now, std) + mean

        # cal water budget
        swvl_prev = math.multiply(pred_prev[:,:3], soil_depth) # (b,4)
        swvl_now = math.multiply(pred_now[:,:3], soil_depth) # (b,4)
        delta_swvl = math.reduce_sum(swvl_now-swvl_prev, axis=-1) #(b,)
        w_b = aux-delta_swvl-pred_now[:,-2]-pred_now[:,-1] #(b,)

        # cal ratio and distribute
        pred_new = []
        w_a = math.reduce_sum(swvl_now, axis=-1) #(b)

        #1. give to three soil layers
        for i in range(3):
            ratio = math.divide(swvl_now[:,i], w_a)
            water_add = math.multiply(w_b, ratio)
            pred_new.append((water_add+swvl_now[:,i])/soil_depth[i]) 
        pred_new.append(pred_now[:,-2])
        pred_new.append(pred_now[:,-1])

        """2. give to third soil layers
        pred_new.append(swvl_now[:,0]/soil_depth[0])
        pred_new.append(swvl_now[:,1]/soil_depth[1])
        pred_new.append((swvl_now[:,2]+w_b)/soil_depth[2])

        pred_new.append(pred_now[:,-2])
        pred_new.append(pred_now[:,-1]) 
        """
        """
        #3. give to all variables
        w_c = math.reduce_sum(pred_now[:,-2:],axis=-1) + w_a
        for i in range(3):
            ratio = math.divide(swvl_now[:,i], w_c)
            water_add = math.multiply(w_b, ratio)
            pred_new.append((water_add+swvl_now[:,i])/soil_depth[i])         
        pred_new.append(pred_now[:,-2]+math.multiply(w_b, math.divide(pred_now[:,-2], w_c)))
        pred_new.append(pred_now[:,-1]+math.multiply(w_b, math.divide(pred_now[:,-1], w_c)))
        """
        pred_new = tf.stack(pred_new, axis=-1) #(b,6)
        pred_new = math.divide(pred_new-mean, std)
        pred = tf.stack([pred_prev_save, pred_new], axis=1) #(b,2,6)
        return pred


class HardMTLModel_v2(Model):
    """multitasks model with hard physical constrain through residual layer."""

    def __init__(self, cfg):
        super().__init__()
        self.num_out = cfg["num_out"]
        self.shared_layer = LSTM(cfg["hidden_size"],return_sequences=True)
        self.drop = Dropout(cfg["dropout_rate"])
        self.head_layers_prev = []
        for i in range(cfg["num_out"]):
            self.head_layers_prev.append(Dense(1, name='head_layer_prev_'+str(i+1)))
        self.head_layers_now = []
        for i in range(cfg["num_out"]-1):
            self.head_layers_now.append(Dense(1, name='head_layer_now_'+str(i+1)))
        self.resid_layer = MassConsLayer(cfg)

    def call(self, inputs, aux, mean, std):
        x = self.shared_layer(inputs)
        x = self.drop(x, training=True)
        pred_prev = []
        for i in range(self.num_out):
            pred_prev.append(self.head_layers_prev[i](x[:,-2])) #(b,6)
        pred_now = []
        for i in range(self.num_out-1):
            pred_now.append(self.head_layers_now[i](x[:,-1])) #(b,5)

        pred_prev = tf.concat(pred_prev, axis=-1) #(b,6)
        pred_now = tf.concat(pred_now, axis=-1) #(b,5)
        pred = self.resid_layer(pred_prev, pred_now, aux, mean, std) #(b,2,6)
        return pred


class HardMTLModel_v3(Model):
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
        x = self.drop(x, training=True)
        pred = []
        for i in range(self.num_out):  # each heads
            pred.append(self.head_layers[i](x[:,-2:]))
        pred = tf.concat(pred, axis=-1) #(b, 2, 6)
        pred_ = pred

        aux = tf.cast(aux, 'float32')
        pred = tf.cast(pred, 'float32')
        std = tf.cast(std, 'float32')
        mean = tf.cast(mean, 'float32')
        soil_depth = [70, 210, 720] # mm 
        pred_prev_save, pred_now = pred[:,0], pred[:,1]  #(b,6)

        pred_prev = math.multiply(pred_prev_save, std) + mean
        pred_now = math.multiply(pred_now, std) + mean

        # cal water budget
        swvl_prev = math.multiply(pred_prev[:,:3], soil_depth) # (b,4)
        swvl_now = math.multiply(pred_now[:,:3], soil_depth) # (b,4)
        delta_swvl = math.reduce_sum(swvl_now-swvl_prev, axis=-1) #(b,)
        w_b = aux-delta_swvl-pred_now[:,-2]-pred_now[:,-1] #(b,)

        # cal ratio and distribute
        pred_new = []
        w_a = math.reduce_sum(swvl_now, axis=-1) #(b)

        for i in range(3):
            ratio = math.divide(swvl_now[:,i], w_a)
            water_add = math.multiply(w_b, ratio)
            pred_new.append((water_add+swvl_now[:,i])/soil_depth[i]) 
        pred_new.append(pred_now[:,-2])
        pred_new.append(pred_now[:,-1])

        """
        w_c = math.reduce_sum(pred_now[:,-2:],axis=-1) + w_a
        for i in range(3):
            ratio = math.divide(swvl_now[:,i], w_c)
            water_add = math.multiply(w_b, ratio)
            pred_new.append((water_add+swvl_now[:,i])/soil_depth[i])         
        pred_new.append(pred_now[:,-2]+math.multiply(w_b, math.divide(pred_now[:,-2], w_c)))
        pred_new.append(pred_now[:,-1]+math.multiply(w_b, math.divide(pred_now[:,-1], w_c)))
        """
        
        pred_new = tf.stack(pred_new, axis=-1) #(b,6)
        pred_new = math.divide(pred_new-mean, std)
        pred = tf.stack([pred_prev_save, pred_new], axis=1) #(b,2,6)
        return pred_, pred


