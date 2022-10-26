import time
import os

from tqdm import trange
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from sklearn.metrics import r2_score

from data_gen import load_train_data, load_test_data
from loss import PHydroLoss
from model import VanillaLSTM



# NOTE: If we add decorator `tf.function` of `train_step`, and 
#       we try to trained model twice. It will raise error: 
#       "with ValueError: tf.function only supports singleton 
#       tf.Variables created on the first call. Make sure the 
#       tf.Variable is only created once or created outside 
#       tf.function". Thus, `train_step` only used for multi-task 
#       model to speed up trainning. see `train_multi`.
@tf.function 
def train_step(x, y, model,loss_fn, optim, metric):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    metric.update_state(y, pred)


@tf.function
def test_step(x, y, model, metric):
    pred = model(x, training=False)
    metric.update_state(y, pred)



def train_single(x, 
                 y, 
                 cfg, 
                 num_task, 
                 valid_split=None,
                 num_repeat=None):
    # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively 
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`. 
    optim = Adam(cfg["learning_rate"])
    loss_fn = PHydroLoss(cfg)
    metric = tfa.metrics.RSquare()
    patience = 5
    wait = 0
    best = -9999
    flag = 1
    # prepare for model with different seeds
    save_folder = cfg["outputs_path"]+"saved_model/"+\
        cfg["model_name"]+'/'+str(num_task)+'/'+str(num_repeat)+'/'
        
    # Prepare for validate
    if valid_split:
        nt = x.shape[1]
        N = int(nt*valid_split)
        x_valid, y_valid = x[:,N:], y[:,N:]
        x, y = x[:,:N], y[:,:N]
        x_valid, y_valid = load_test_data(x_valid, y_valid, cfg["seq_len"])
        valid_metric = tfa.metrics.RSquare()

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for _ in range(100):
        model = VanillaLSTM(cfg)
        with trange(1, cfg["epochs"]+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg["model_name"]+' '+str(num_task)+' member '+str(num_repeat))
                
                # train
                t0 = time.time()
                for iter in range(0, cfg["niter"]): 
                    x_batch, y_batch = load_train_data(x, y, cfg)
                    with tf.GradientTape() as tape:
                        pred = model(x_batch, training=True)
                        loss = loss_fn(y_batch, pred)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optim.apply_gradients(zip(grads, model.trainable_variables))
                    metric.update_state(y_batch, pred)
                train_acc = metric.result().numpy()
                metric.reset_states()
                t1 = time.time()

                # log
                loss_str = "Epoch {} Train NSE {:.3f} time {:.2f}".format(
                    epoch, train_acc, t1-t0)
                print(loss_str)
                pbar.set_postfix(loss=train_acc)

                # refresh train if loss equal to NaN
                # Will build fresh model and re-train it
                # until it didn't have NaN loss.
                if np.isnan(train_acc):
                    break

                # validate
                if valid_split:
                    if epoch % 20 == 0:
                        wait += 1

                        # NOTE: We use larger than 3 years for validate, and 
                        #       used grids-mean NSE as valid metrics.
                        #       We cannot use `tf.data.Dataset.from_tensor_slices`
                        #       to transform nd.array to tensor. Because it will
                        #       put all valid data into GPU, which exceed memory.
                        t0 = time.time()
                        for i in range(x_valid.shape[0]):
                            pred = model(x_valid[i], training=False)
                            valid_metric.update_state(y_valid[i], pred)
                        val_acc = valid_metric.result().numpy()
                        valid_metric.reset_states()
                        t1 = time.time()

                        # log in `red`
                        loss_str = '\033[1;31m%s\033[0m' % \
                            "Epoch {} Val NSE {:.3f} time {:.2f}".format(
                                epoch, val_acc, t1-t0)
                        print(loss_str)

                        # save best model by val loss
                        if val_acc > best:
                            model.save_weights(save_folder)
                            wait = 0 # release wait
                            best = val_acc
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if train_acc > best:
                        best = train_acc
                        wait = 0
                        model.save_weights(save_folder)
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                
                # early stopping
                if wait >= patience:
                    return
            return
            


