
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import trange
import argparse
from pathlib import Path, PosixPath
import logging
import datetime, time

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adadelta, Adam
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from src.loss import MSELoss
from src.model import MTLHydro, LSTMModel, MTLLSTM
from src.metrics import RMetrics, MTLRSquareMetrics
from src.data import Dataset


def get_variables(trainable_variables, name):
    return [v for v in trainable_variables if name in v.name]

#def combine_gradients_list(main_grads, aux_grads, lamb=1):
#    return [main_grads[i] + lamb * aux_grads[i] for i in range(len(main_grads))]


def combine_gradients_list(shared_gradient):
    a0 = shared_gradient[0]
    a1 = shared_gradient[1]
    a2 = shared_gradient[2]
    a3 = shared_gradient[3]
    a4 = shared_gradient[4]
    a5 = shared_gradient[5]
    return [a0[i]+a1[i]+a2[i]+a3[i]+a4[i]+a5[i] for i in range(len(a0))]



@tf.function
def train_step(cfg, X, y, model, loss_fn, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(X, training=True)
        loss = []
        for i in range(cfg["num_out"]):
            loss.append(loss_fn(y[:,i], pred[:,i]))

    trainable_vars = model.trainable_variables
    shared_vars = get_variables(trainable_vars, "shared_layer")
    vars = []
    for i in range(cfg["num_out"]):
        vars.append(get_variables(trainable_vars, "head_"+str(i+1)))
    
    gradient = []
    shared_gradient = []
    for i in range(cfg["num_out"]):
        gradient.append(tape.gradient(loss[i], vars[i]))
        shared_gradient.append(tape.gradient(loss[i], shared_vars))

    combined_gradient = combine_gradients_list(shared_gradient)

    for i in range(cfg["num_out"]):
        optimizer.apply_gradients(zip(gradient[i], vars[i]))
    optimizer.apply_gradients(zip(combined_gradient, shared_vars))
    return {"loss_1": loss[0], 
            "loss_2": loss[1], 
            "loss_3": loss[2], 
            "loss_4": loss[3], 
            "loss_5": loss[4],
            "loss_6": loss[5]}, pred
