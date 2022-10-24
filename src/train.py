import time

from tqdm import trange
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adagrad
import tensorflow_addons as tfa
from sklearn.metrics import r2_score

from data_generator import load_data, reverse_normalize
from loss import PHydroLoss
from model import VanillaLSTM


@tf.function # for speed up
def train_step(x, y, model,loss_fn, optim, metric):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    metric.update_state(y, pred)


@tf.function
def test_step(x, model):
    pred = model(x, training=False)
    return pred


def train(model, x, y, cfg, i=None, validation_split=None):
    optim = Adagrad(cfg["learning_rate"])
    loss_fn = PHydroLoss(cfg)
    metric = tfa.metrics.RSquare()
    patience = 20
    wait = 0
    best = 0
    
    if validation_split:
        N = int(x.shape[1]*validation_split)
        x_valid, y_valid = x[:,N:], y[:,N:]
        x, y = x[:,:N], y[:,N]
        valid_metric = tfa.metrics.RSquare()

    save_folder = cfg["outputs_path"]+"saved_model/"+cfg["model_name"]+'/'
    if cfg["model_name"] == 'single_task': save_folder = save_folder+str(i)+'/'

    with trange(1, cfg["epochs"]+1) as pbar:
        for epoch in pbar:
            pbar.set_description("Training {}".format(cfg["model_name"]))
            
            t0 = time.time()
            #TODO(lilu) Adaptively iteration step setting.
            for iter in range(0, cfg["niter"]): 
                x_batch, y_batch = load_data(x, y, cfg)
                train_step(x_batch, y_batch, model, loss_fn, optim, metric)
            train_acc = metric.result().numpy()
            metric.reset_states()
            t1 = time.time()

            loss_str = "Epoch {} Loss {:.3f} time {:.2f}".format(epoch, train_acc, t1-t0)
            print(loss_str)
            pbar.set_postfix(loss=train_acc)

            #TODO(lilu)save best strategy 
            if validation_split:
                pass
            else:
                # save each 100 epoch
                if epoch % 100 == 0:
                    model.save_weights(save_folder)
                
            #TODO(lilu)early stopping strategy
            wait += 1
            if train_acc > best:
                best = train_acc
                wait = 0
            if wait >= patience:
                break
            


def predict(x, y, scaler, cfg):
    # 
    print(x.shape, y.shape) #(ngrid, nt, seq_len, 9) (ngrid,nt,6)
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"]) #(1, ngrid, 6)

    # save folder
    save_folder = cfg["outputs_path"]+"saved_model/"+cfg["model_name"]+'/'

    y_pred = []
    for j in range(1): # for each feat
        print(j)
        if cfg["model_name"] == 'single_task': 
            save_folder = save_folder+str(j)+'/'
        model = VanillaLSTM(cfg)
        model.load_weights(save_folder)  

        t0 = time.time()  
        tmp = []
        r2 = []
        for i in range(x.shape[0]):  # for each grids (samples,seq_len,nfeat)
            pred = test_step(x[i], model) # (nt, 1)
            pred = pred*std[:,i,j]+mean[:,i,j] 
            tmp.append(pred)
            r2.append(r2_score(y[i,:,j],pred[:,0]))
        mean_r2 = np.nanmean(np.array(r2))
        t1 = time.time()
        print("Var {} Mean NSE {:.3f} Time {:.2f}".format(j, mean_r2, t1-t0))
        tmp = np.stack(tmp, axis=0)
        y_pred.append(tmp)
    return np.concatenate(y_pred, axis=-1)
