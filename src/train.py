import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from data_gen import load_test_data, load_train_data
from loss import MassConsLoss, NaNMSELoss
from metric import NSEMetrics
from model import HardMTLModel_v1, MTLModel_v1, MTLModel_v2, STModel


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
    best = -9999

    # prepare save folder for models with different seeds
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
            cfg, x_valid, y_valid, aux_valid, scaler, stride=4)
        valid_metric = NSEMetrics(cfg)

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for _ in range(100):
        # prepare models
        if cfg["model_name"] in ['single_task']:
            model = STModel(cfg)
        elif cfg["model_name"] in ['multi_tasks_v1', 'soft_multi_tasks']:
            model = MTLModel_v1(cfg)
        elif cfg["model_name"] in ['multi_tasks_v2']:
            model = MTLModel_v2(cfg)
        elif cfg["model_name"] in ['hard_multi_tasks_v1']:
            model = HardMTLModel_v1(cfg)

        with trange(1, cfg["epochs"]+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg["model_name"]+' '+str(num_task)+' member '+str(num_repeat))

                t_begin = time.time()
                # train
                MCLoss, MSELoss = 0, 0
                for iter in range(0, cfg["niter"]):
                    # generate batch data
                    x_batch, y_batch, aux_batch, mean_batch, std_batch = \
                        load_train_data(cfg, x, y, aux, scaler)
                    with tf.GradientTape() as tape:
                        # cal MSE loss
                        if cfg["model_name"] in \
                            ['single_task','multi_tasks_v1','soft_multi_tasks']:
                            pred = model(x_batch, training=True)
                        elif cfg["model_name"] in ['multi_tasks_v2']:
                            pred, adaptive_loss = model(x_batch, y_batch, training=True)  
                        else:
                            pred = model(x_batch, aux_batch, mean_batch, std_batch, training=True)
                        mse_loss = NaNMSELoss(cfg)(y_batch, pred[:,1]) # only cal on last step
                        MSELoss+=mse_loss

                        # cal physic loss
                        if cfg["model_name"] != 'single_task':
                            phy_loss = MassConsLoss(cfg, mean_batch, std_batch)(aux_batch, pred)
                            MCLoss += phy_loss

                        # cal all loss
                        if cfg["model_name"] in ["soft_multi_tasks"]:
                            loss = (1-cfg["alpha"])*mse_loss + (cfg["alpha"])*phy_loss
                        elif cfg["model_name"] in ["multi_tasks_v2"]:
                            loss = adaptive_loss
                        else:
                            loss = mse_loss
                    # gradient tape
                    grads = tape.gradient(loss, model.trainable_variables)
                    optim.apply_gradients(zip(grads, model.trainable_variables))
                    metric.update_state(y_batch, pred[:,1])
                t_end = time.time()

                # get loss log
                t_acc = metric.result()
                if cfg["model_name"] == 'single_task':
                    t0 = t_acc["loss"].numpy()
                    loss_str = "Epoch {} Train NSE {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,t0,MSELoss/cfg["niter"],MCLoss/cfg["niter"],t_end-t_begin)
                else:
                    t1,t2,t3,t4,t5,t6 = t_acc["0"].numpy(),t_acc["1"].numpy(),t_acc["2"].numpy(), \
                        t_acc["3"].numpy(),t_acc["4"].numpy(),t_acc["5"].numpy()
                    loss_str = "Epoch {} Train NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} SWVL4 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(
                        epoch,t1,t2,t3,t4,t5,t6,MSELoss/cfg["niter"],MCLoss/cfg["niter"],t_end-t_begin)
                print(loss_str)
                metric.reset_states()

                # refresh train if loss equal to NaN. Will build fresh model and 
                # re-train it until it didn't have NaN loss.
                if np.isnan(MSELoss):
                    break

                # validate
                if valid_split:
                    MC_valid_loss, MSE_valid_loss = 0, 0
                    if epoch % 20 == 0:
                        wait += 1

                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
                        for i in range(x_valid.shape[0]):
                            if cfg["model_name"] in ['single_task', \
                                'multi_tasks_v1', 'multi_tasks_v2', 'soft_multi_tasks']:
                                pred = model(x_valid[i], training=False)
                            else:
                                pred = model(x_valid[i], aux_valid[i], \
                                    mean_valid[i], std_valid[i], training=False)
                            # cal mse loss
                            mse_valid_loss = NaNMSELoss(cfg)(y_valid[i], pred[:,1])
                            MSE_valid_loss+=mse_valid_loss
                            valid_metric.update_state(y_valid[i], pred[:,1])

                            # cal phy loss
                            phy_loss = MassConsLoss(
                                cfg, mean_valid[i], std_valid[i])(aux_valid[i], pred)
                            MC_valid_loss+=phy_loss
                        t_end = time.time()

                        # get loss log
                        v_acc = valid_metric.result()
                        if cfg["model_name"] == 'single_task':
                            val = v_acc["loss"].numpy()
                            loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val NSE {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,val,MSE_valid_loss/x_valid.shape[0], 
                                    MC_valid_loss/x_valid.shape[0],t_end-t_begin)
                            val_save_acc = val
                        else:
                            t1,t2,t3,t4,t5,t6 = v_acc["0"].numpy(),v_acc["1"].numpy(),\
                                v_acc["2"].numpy(), v_acc["3"].numpy(),v_acc["4"].numpy(),\
                                    v_acc["5"].numpy()
                            loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} SWVL4 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(epoch,t1,t2,t3,t4,t5,t6, MSE_valid_loss/x_valid.shape[0], 
                                    MC_valid_loss/x_valid.shape[0],t_end-t_begin)
                            val_save_acc = (t1+t2+t3+t4+t5+t6)/6
                        print(loss_str)
                        valid_metric.reset_states()

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc > best:
                        #if MSE_valid_loss < best:
                            model.save_weights(save_folder)
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
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