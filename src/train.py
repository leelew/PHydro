import time

from tqdm import trange
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adagrad
import tensorflow_addons as tfa
from sklearn.metrics import r2_score

from data_gen import load_data, load_test_data, reverse_normalize
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



def train_single(model, 
                 x, 
                 y, 
                 cfg, 
                 i, 
                 valid_split=None):
    # Prepare for training
    # Only use `Adagrad` in this study.  
    optim = Adagrad(cfg["learning_rate"])
    loss_fn = PHydroLoss(cfg)
    metric = tfa.metrics.RSquare()
    patience = 10
    wait = 0
    best = 0
    save_folder = cfg["outputs_path"]+"saved_model/"+\
        cfg["model_name"]+'/'+str(i)+'/'
    
    # Prepare for validate
    if valid_split:
        nt = x.shape[1]
        N = int(nt*valid_split)
        x_valid, y_valid = x[:,N:], y[:,N:]
        x, y = x[:,:N], y[:,:N]
        x_valid, y_valid = load_test_data(x_valid, y_valid, cfg["seq_len"])
        val_data = tf.data.Dataset.from_tensor_slices((
            x_valid.reshape(-1, cfg["seq_len"], cfg["num_feat"]),
            y_valid.reshape(-1, 1)))
        nsample = x_valid.shape[1]
        val_data = val_data.batch(nsample)
        valid_metric = tfa.metrics.RSquare()

    # train and validate
    # NOTE: We preprare three callbacks for training,
    #       `Adadelta` for adaptively learning rate, 
    #       early stopping and save best model.
    with trange(1, cfg["epochs"]+1) as pbar:
        for epoch in pbar:
            pbar.set_description("Training {}".format(cfg["model_name"]))
            
            # train
            t0 = time.time()
            #TODO(lilu) Adaptively iteration step setting.
            for iter in range(0, cfg["niter"]): 
                x_batch, y_batch = load_data(x, y, cfg)
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
            loss_str = "Epoch {} Train Loss {:.3f} time {:.2f}".format(epoch, train_acc, t1-t0)
            print(loss_str)
            pbar.set_postfix(loss=train_acc)
            wait += 1

            # validate
            if valid_split:
                if epoch % 100 == 0:
                    # NOTE: We use larger than 3 years for validate, and we 
                    #       calculate mean R2 of all avaliable grids to 
                    #       determine which model is better.
                    t0 = time.time()
                    for x_batch_val, y_batch_val in val_data:
                        pred = model(x_batch_val, training=False)
                        valid_metric.update_state(y_batch_val, pred)
                    val_acc = valid_metric.result().numpy()
                    valid_metric.reset_states()
                    t1 = time.time()

                    # log
                    loss_str = "Epoch {} Val Loss {:.3f} time {:.2f}".format(epoch, val_acc, t1-t0)
                    print(loss_str)

                    # save best model/early stopping according to val loss
                    if val_acc > best:
                        model.save_weights(save_folder)
                        wait = 0 # release wait
                        best = val_acc
                        print(f'Save Epoch {epoch} Model' )
            else:
                # save each 100 epoch
                if epoch % 100 == 0:
                    model.save_weights(save_folder)
                    print(f'Save Epoch {epoch} Model' )
                # early stopping according to train loss
                if train_acc > best:
                    best = train_acc
                    wait = 0
            
            """
            # early stopping
            if wait >= patience:
                break
            """

def predict(x, y, scaler, cfg):
    # 
    print(x.shape, y.shape) #(ngrid, nt, seq_len, 9) (ngrid,nt,6)
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"]) #(1, ngrid, 6)

    # save folder
    save_folder = cfg["outputs_path"]+"saved_model/"+cfg["model_name"]+'/'

    y_pred = []
    for j in range(5,6): # for each feat
        print(j)
        if cfg["model_name"] == 'single_task': 
            save_folder = save_folder+str(j)+'/'
        model = VanillaLSTM(cfg)
        model.load_weights(save_folder)  

        t0 = time.time()  
        tmp = []
        r2 = []
        for i in range(x.shape[0]):  # for each grids (samples,seq_len,nfeat)
            print(i)
            pred = model(x[i], training=False) # (nt, 1)
            pred = pred*std[:,i,j]+mean[:,i,j] 
            tmp.append(pred)
            r2.append(r2_score(y[i,:,j],pred[:,0]))
        mean_r2 = np.nanmean(np.array(r2))
        t1 = time.time()

        print("Var {} Mean NSE {:.3f} Time {:.2f}".format(j, mean_r2, t1-t0))
        tmp = np.stack(tmp, axis=0)
        y_pred.append(tmp)
    return np.concatenate(y_pred, axis=-1)
