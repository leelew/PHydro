import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from data_gen import load_test_data, load_train_data, make_train_data
from loss import MassConsLoss, WeightedMSELoss
from metric import NSEMetrics
from model import (HardMTLModel_v1, HardMTLModel_v2, HardMTLModel_v3,MTLModel_v1, MTLModel_v2,
                   STModel)


def train(x,
          y,
          aux,
          scaler,
          cfg,
          num_repeat,
          num_task=None,
          valid_split=True):
    # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
    optim = Adam(cfg["learning_rate"])
    metric = NSEMetrics(cfg)
    patience = 10
    wait = 0
    best = 9999999999

    # Prepare save folder for models with different seeds
    if cfg["model_name"] == 'single_task':
        save_folder = cfg["outputs_path"]+"saved_model/" +\
            cfg["model_name"]+'/'+str(num_task)+'/'+str(num_repeat)+'/'
    else:
        save_folder = cfg["outputs_path"]+"saved_model/" +\
            cfg["model_name"]+'/'+str(num_repeat)+'/'

    # Prepare for validate
    if valid_split:
        nt = x.shape[1]
        N = int(nt*cfg["split_ratio"])
        x_valid, y_valid, aux_valid = x[:, N:], y[:, N:], aux[:, N:]
        x, y, aux = x[:, :N], y[:, :N], aux[:, :N]
        x_valid, y_valid, aux_valid, mean_valid, std_valid = load_test_data(
            cfg, x_valid, y_valid, aux_valid, scaler, stride=1)
        valid_metric = NSEMetrics(cfg)
    print(np.isnan(x_valid).any(), np.isnan(y_valid).any())
    
    # directly make all possible inputs
    x_train, y_train, aux_train, mean_train, std_train = make_train_data(cfg, x, y, aux, scaler, stride=50)
    print(x_train.shape, y_train.shape, aux_train.shape, mean_train.shape, std_train.shape)

    if cfg["random_ratio"] != 1:            
        if (cfg["spatial_cv"] == -1) and (cfg["temporal_cv"] == -1):
            a = np.load(cfg["inputs_path"]+ \
                "random_index_{ratio}.npy".format(ratio=cfg['random_ratio']))
        elif cfg["temporal_cv"] != -1:
            a = np.load(cfg["inputs_path"]+ \
                "random_index_{ratio}_tcv.npy".format(ratio=cfg['random_ratio']))
        else:
            a = np.load(cfg["inputs_path"]+ \
                "random_index_{ratio}_cv.npy".format(ratio=cfg['random_ratio']))
        x_train = x_train[a]
        y_train = y_train[a]
        aux_train = aux_train[a]
        mean_train = mean_train[a]
        std_train = std_train[a]
        
    print(x_train.shape, y_train.shape, aux_train.shape, mean_train.shape, std_train.shape)
    train_df = tf.data.Dataset.from_tensor_slices((x_train, y_train, aux_train, mean_train, std_train))
    train_df = train_df.shuffle(x_train.shape[0]).batch(cfg["batch_size"])

    # Train and validate
    # NOTE: We preprare two callbacks for training: early stopping and save best model.
    for _ in range(100):
        # prepare models
        if cfg["model_name"] in ['single_task']:
            model = STModel(cfg) # single-task model
        elif cfg["model_name"] in ['multi_tasks_v1','soft_multi_tasks']:
            model = MTLModel_v1(cfg) # standard multi-tasks model
        elif cfg["model_name"] in ['multi_tasks_v2']:
            model = MTLModel_v2(cfg) # uncertainty weighted multi-tasks model
        elif cfg["model_name"] in ['hard_multi_tasks_v1']:
            model = HardMTLModel_v1(cfg) # redistribution hard constrain model
        elif cfg["model_name"] in ["hard_multi_tasks_v2"]:
            model = HardMTLModel_v2(cfg) # residual hard constrain model
        elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
            model = HardMTLModel_v3(cfg) # residual hard constrain model

        with trange(1, cfg["epochs"]+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg["model_name"]+' '+str(num_task)+' member '+str(num_repeat))

                t_begin = time.time()
                # train
                MCLoss, MSELoss, all_loss = 0, 0, 0
                for step, (x_batch, y_batch, aux_batch, mean_batch, std_batch) in enumerate(train_df):
                
                    # discard iteration when NaN in observation
                    if cfg["model_name"] in ["hard_multi_tasks_v1", "hard_multi_tasks_v2","hard_multi_tasks_v3"]:
                        if np.isnan(y_batch).any():
                            continue

                    with tf.GradientTape() as tape:
                        # predict by models
                        if cfg["model_name"] in \
                            ['single_task','multi_tasks_v1','soft_multi_tasks']:
                            pred = model(x_batch, training=True)
                        elif cfg["model_name"] in ['multi_tasks_v2']:
                            pred, adaptive_loss = model(x_batch, y_batch, training=True) 
                        elif cfg["model_name"] in ["hard_multi_tasks_v1","hard_multi_tasks_v2"]:
                            pred = model(x_batch, aux_batch, mean_batch, std_batch, training=True)
                        elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
                            pred, pred_ = model(x_batch, aux_batch, mean_batch, std_batch, training=True)

                        # cal physic loss
                        # NOTE: single task model cannot cal water budget during training.
                        if cfg["model_name"] != 'single_task':
                            phy_loss = MassConsLoss(cfg, mean_batch, std_batch)(aux_batch, pred)
                            MCLoss += phy_loss

                        # cal MSE loss
                        mse_loss = WeightedMSELoss(cfg)(y_batch, pred)
                        MSELoss+=mse_loss

                        if cfg["model_name"] == 'hard_multi_tasks_v3':
                            mse_loss_ = WeightedMSELoss(cfg)(y_batch, pred_)

                        # cal combine loss if necessary
                        if cfg["model_name"] in ["soft_multi_tasks"]:
                            loss = mse_loss + (cfg["alpha"])*phy_loss
                        elif cfg["model_name"] in ["multi_tasks_v2"]:
                            loss = adaptive_loss
                        elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
                            loss = mse_loss + (cfg["alpha"])*phy_loss #+ mse_loss_
                        else:
                            loss = mse_loss
                        all_loss+=loss

                    # gradient tape
                    grads = tape.gradient(loss, model.trainable_variables)
                    optim.apply_gradients(zip(grads, model.trainable_variables))
                    metric.update_state(y_batch, pred)
                t_end = time.time()

                # get loss log
                t_acc = metric.result()
                if cfg["model_name"] == 'single_task':
                    t0 = t_acc["loss"].numpy()
                    loss_str = "Epoch {} Train NSE {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,t0,MSELoss/cfg["niter"],MCLoss/cfg["niter"],t_end-t_begin)
                else:
                    t1,t2,t3,t4,t5 = t_acc["0"].numpy(),t_acc["1"].numpy(),t_acc["2"].numpy(), \
                        t_acc["3"].numpy(),t_acc["4"].numpy()
                    loss_str = "Epoch {} Train NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(
                        epoch,t1,t2,t3,t4,t5,MSELoss/cfg["niter"],MCLoss/cfg["niter"],t_end-t_begin)
                print(loss_str)
                metric.reset_states()

                # refresh train if loss equal to NaN. Will build fresh model and 
                # re-train it until it didn't have NaN loss.
                if np.isnan(MSELoss):
                    break

                # validate
                if valid_split:
                    MC_valid_loss, MSE_valid_loss, all_loss = 0, 0, 0
                    if epoch % 10 == 0:
                        wait += 1

                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
                        for i in range(x_valid.shape[0]):
                            # predict at each grids
                            if cfg["model_name"] in ['hard_multi_tasks_v1','hard_multi_tasks_v2']:
                                pred = model(x_valid[i], aux_valid[i], \
                                    mean_valid[i], std_valid[i], training=False)
                            elif cfg["model_name"] in ['multi_tasks_v2']:
                                pred, adap_val_loss = model(x_valid[i], y_valid[i], training=True) 
                            elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
                                pred_,pred = model(x_valid[i], aux_valid[i], \
                                    mean_valid[i], std_valid[i], training=False)
                            else:
                                pred = model(x_valid[i], training=False)
                            
                            # cal mse loss
                            mse_valid_loss = WeightedMSELoss(cfg)(y_valid[i], pred)
                            MSE_valid_loss+=mse_valid_loss

                            # cal phy loss
                            phy_valid_loss = MassConsLoss(
                                cfg, mean_valid[i], std_valid[i])(aux_valid[i], pred)
                            MC_valid_loss+=phy_valid_loss

                            if cfg["model_name"] in ["hard_multi_tasks_v3"]:
                                phy_valid_loss_ = MassConsLoss(
                                    cfg, mean_valid[i], std_valid[i])(aux_valid[i], pred_)
                                mse_valid_loss_ = WeightedMSELoss(cfg)(y_valid[i], pred_)

                            # cal combine loss if necessary
                            if cfg["model_name"] in ["soft_multi_tasks"]:
                                loss = mse_valid_loss + \
                                    (cfg["alpha"])*phy_valid_loss
                                loss = mse_valid_loss
                            elif cfg["model_name"] in ["multi_tasks_v2"]:
                                loss = adap_val_loss
                            elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
                                loss =  mse_valid_loss_ + \
                                        (cfg["alpha"])*phy_valid_loss_ #+ mse_valid_loss
                            else:
                                loss = mse_valid_loss
                            all_loss+=loss

                            # update valid metric
                            valid_metric.update_state(y_valid[i], pred)
                        t_end = time.time()

                        # get loss log
                        v_acc = valid_metric.result()
                        if cfg["model_name"] == 'single_task':
                            val = v_acc["loss"].numpy()
                            loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val NSE {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,val,MSE_valid_loss/x_valid.shape[0],MC_valid_loss/x_valid.shape[0],t_end-t_begin)
                        else:
                            t1,t2,t3,t4,t5 = v_acc["0"].numpy(),v_acc["1"].numpy(),\
                                v_acc["2"].numpy(), v_acc["3"].numpy(),v_acc["4"].numpy()
                            loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,t1,t2,t3,t4,t5,MSE_valid_loss/x_valid.shape[0],MC_valid_loss/x_valid.shape[0],t_end-t_begin)
                        print(loss_str)
                        valid_metric.reset_states()

                        # save best model by val loss
                        if all_loss/x_valid.shape[0] < best:
                            model.save_weights(save_folder)
                            wait = 0  # release wait
                            best = all_loss/x_valid.shape[0]
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if all_loss < best:
                        best = all_loss
                        wait = 0
                        model.save_weights(save_folder)
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    return
            return


# NOTE: If we add decorator `tf.function` of `train_step`, and
#       we try to trained model twice. It will raise error:
#       "with ValueError: tf.function only supports singleton
#       tf.Variables created on the first call. Make sure the
#       tf.Variable is only created once or created outside
#       tf.function". Thus, `train_step` only used for multi-task
#       model to speed up trainning. see `train_multi`.
@tf.function
def train_step(x, y, model, loss_fn, optim, metric):
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
