import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam


def train_shared_layer(X, y, cfg, model, loss_fn, optimizer):
    for epoch in range(cfg["epochs"]):
        for iter in range(cfg["niters"]):
            x_batch, y_batch = make_batch_data(X, y)
            with tf.GradientTape(persistent=True) as tape:
                pred = model(x_batch)
                loss = loss_fn(y_batch, x_batch)
            train_vars = model.trainable_variables
            grads = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))
    return model

    # FIXME: save models


def train_head_layer(cfg):
    # train head layers
    # ensure not single-task
    if cfg["is_tune"] and (cfg["model_name"] is not "singlevar"):
        for epoch in range(cfg["epochs_ft"]):
            for iter in range(cfg["niters_ft"]):
                x_batch, y_batch = make_batch_data(x_train, y_train)
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


def train_singletask(X, y, cfg, model, loss_fn, optimizer,):
    a = []
    for i in range(cfg["num_out"]):
        mdl = train(X[:, :, i], y[:, :, i], cfg, model, loss_fn, optimizer)
        a.append(mdl)
    return a


def train_multitask():
    pass


def predict_singletask():
    pass


def predict_multitask():
    pass
