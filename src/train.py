import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam


def train_shared_layer(X, y, cfg, model, loss_fn, optimizer, data_generator):
    """train shared LSTM layers"""
    for epoch in range(cfg["epochs"]):
        for iter in range(cfg["niters"]):
            x_batch, y_batch = data_generator(X, y)
            with tf.GradientTape(persistent=True) as tape:
                pred = model(x_batch)
                loss = loss_fn(y_batch, x_batch)
            train_vars = model.trainable_variables
            grads = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))
    return model    # FIXME: save models #TODO: Add validate process


def train_head_layer(X, y, cfg, model, loss_fn, optimizer, data_generator):
    """train head dense layers for multi-tasks"""
    def get_variables(trainable_variables, name):
        return [v for v in trainable_variables if name in v.name]

    # ensure not single task and needs tune for multi-tasks
    if cfg["is_tune"] and (cfg["model_name"] != "single_task"):
        for epoch in range(cfg["epochs_ft"]):
            for iter in range(cfg["niters_ft"]):
                x_batch, y_batch = data_generator(X, y)
                with tf.GradientTape(persistent=True) as tape:
                    pred = model(x_batch)
                    loss = []
                    for i in range(cfg["num_out"]):
                        loss.append(loss_fn(y_batch[:, i], pred[:, i]))

                trainable_vars = model.trainable_variables
                for i in range(cfg["num_out"]):
                    vars = get_variables(trainable_vars, "head_"+str(i+1))
                    grads = tape.gradient(loss[i], vars)
                    optimizer.apply_gradients(zip(grads, vars))
    return model


def train_single_task(X, y, cfg, model, loss_fn, optimizer, data_generator):
    model_list = []
    for i in range(cfg["num_out"]):
        mdl = train_shared_layer(
            X, y[:, :, i], cfg, model, loss_fn, optimizer, data_generator)
        model_list.append(mdl)
    return model_list


def train_multi_tasks(X, y, cfg, model, loss_fn, optimizer, data_generator):
    model = train_shared_layer(
        X, y, cfg, model, loss_fn, optimizer, data_generator)
    model = train_head_layer(
        X, y, cfg, model, loss_fn, optimizer, data_generator)
    return model


def predict_single_task(X, model_list):
    # (ngrids, samples, seq_len, nfeat)
    y_pred = []
    for i in range(X.shape[0]):  # for each grids
        tmp = []
        for mdl in range(model_list):  # for each feat
            tmp.append(mdl.predict(X[i]))
        tmp = np.concatenate(tmp, axis=-1)  # (samples, num_out)
        y_pred.append(tmp)
    y_pred = np.stack(y_pred, axis=0)  # (ngrids, samples, num_out)
    return y_pred


def predict_multi_tasks(X, model):
    y_pred = model.predict(X)  # FIXME: Add save best model and load best model
    return y_pred
