import sys
sys.path.append('../src')

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Dense, Dropout
import tensorflow.keras.backend as K
from src.loss import MSELoss, NNSELoss
from tensorflow.keras.losses import MeanSquaredError


class MTLLSTM(tf.keras.Model):
    """LSTM in multi-task learning structure."""
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.num_out = cfg["num_out"]
        self.is_multivars = cfg["is_multivars"]

        self.lstm = LSTM(cfg["hidden_size"], 
                         return_sequences=False, #FIXME: maybe is wrong
                         name='shared_layer')#,
                         #recurrent_dropout=cfg["dropout_rate"]) 
        self.head = []
        for i in range(self.num_out):
            self.head.append(Dense(1, name='head_'+str(i+1)))

    def call(self, inputs, training=None, mask=None):
        x = self.lstm(inputs)
        pred = []
        for i in range(self.num_out):
            pred.append(self.head[i](x))
        return tf.concat(pred, axis=-1)

    @tf.function
    def train_step(self, data):
        x, y = data

        _ = self(x)
        loss_fn = MeanSquaredError()


        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)

            loss = []
            for i in range(self.num_out):
                loss.append(loss_fn(y[:,i], y_pred[:,i]))
            print(loss)

        trainable_vars = self.trainable_variables
        shared_vars = self._get_variables(trainable_vars, "shared_layer")
        vars = []
        for i in range(self.num_out):
            vars.append(self._get_variables(trainable_vars, "head_"+str(i+1)))
        
        gradient = []
        shared_gradient = []
        for i in range(self.num_out):
            gradient.append(tape.gradient(loss[i], vars[i]))
            shared_gradient.append(tape.gradient(loss[i], shared_vars))

        combined_gradient = self._combine_gradients_list(shared_gradient)

        for i in range(self.num_out):
            self.optimizer.apply_gradients(zip(gradient[i], vars[i]))
        self.optimizer.apply_gradients(zip(combined_gradient, shared_vars))
        return {"loss_1": loss[0], 
                "loss_2": loss[1], 
                "loss_3": loss[2], 
                "loss_4": loss[3], 
                "loss_5": loss[4],
                "loss_6": loss[5],}
 


    def _get_variables(self, trainable_variables, name):
        return [v for v in trainable_variables if name in v.name]

    #def combine_gradients_list(main_grads, aux_grads, lamb=1):
    #    return [main_grads[i] + lamb * aux_grads[i] for i in range(len(main_grads))]


    def _combine_gradients_list(self, shared_gradient):
        a0 = shared_gradient[0]
        a1 = shared_gradient[1]
        a2 = shared_gradient[2]
        a3 = shared_gradient[3]
        a4 = shared_gradient[4]
        a5 = shared_gradient[5]
        return [a0[i]+a1[i]+a2[i]+a3[i]+a4[i]+a5[i] for i in range(len(a0))]

        
            
def LSTMModel(cfg):
    x = tf.keras.Input((cfg["seq_length"], cfg["num_in"]))
    y = LSTM(cfg["hidden_size"], return_sequences=False)(x)
    y = Dropout(cfg["dropout_rate"])(y)
    if cfg["is_multivars"] == 1:
        y = Dense(6, activation='tanh')(y)
    else:
        y = Dense(1, activation='tanh')(y)
    model = tf.keras.Model(x, y)
    model.summary()
    return model


class MTLHydro(tf.keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.max = kwargs['max']
        self.min = kwargs['min']
        self.swvl_prev = kwargs['swvl_prev']

        self.lstm = LSTM(64,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        bias_initializer='zeros',
                        unit_forget_bias=True,
                        kernel_regularizer=None,
                        recurrent_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        recurrent_constraint=None,
                        bias_constraint=None,
                        dropout=0.0,
                        recurrent_dropout=0.0,
                        return_sequences=False,
                        return_state=False,
                        go_backwards=False,
                        stateful=False,
                        time_major=False,
                        unroll=False)
        self.dense = Dense(6, activation='tanh')


    def call(self, inputs, training=None, mask=None):
        y = self.lstm(inputs)
        y = self.dense(y)

        # reverse normalization 
        y = tf.math.multiply(y, (self.max-self.min)) + self.min

        # turn unit to [mm/day]
        # swvl: mm3/mm3
        # et: mm/day
        # runoff: mm/day
        # p: mm/day
        p = inputs[:, -1, 0]
        r = y[:, -1]
        et = y[:, -2]
        swvl = tf.math.reduce_sum(y[:, :4], axis=-1)
        swvl_prev = self.swvl_prev   #FIXME: use predict or true value?

        # cal physical consistency difference
        physical_loss = (swvl-swvl_prev)-(p-r-et)

        return y, physical_loss
        
    
