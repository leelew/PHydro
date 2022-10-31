import time

from tqdm import trange
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

from data_gen import load_train_data, load_test_data
from loss import RMSELoss, MassConserveLoss, RMetrics
from model import MTLHardLSTM, VanillaLSTM, MTLLSTM


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


def train(x,
          y,
          aux,
          scaler,
          cfg,
          num_repeat,
          num_task=None,
          resid_idx=None,
          valid_split=True):
    # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
    optim = Adam(cfg["learning_rate"])
    metric = RMetrics()#tfa.metrics.RSquare()
    patience = 5
    wait = 0
    best = 9999

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
            cfg, x_valid, y_valid, aux_valid, scaler)
        valid_metric = RMetrics()#tfa.metrics.RSquare()

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for _ in range(100):
        # prepare models
        if cfg["model_name"] == 'single_task':
            model = VanillaLSTM(cfg)
        elif cfg["model_name"] in ['multi_tasks', 'soft_multi_tasks']:
            model = MTLLSTM(cfg)
        else:
            model = MTLHardLSTM(cfg)

        with trange(1, cfg["epochs"]+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg["model_name"]+' '+str(num_task)+' member '+str(num_repeat))

                # train
                MCLoss = 0
                MSELoss = 0
                t0 = time.time()
                for iter in range(0, cfg["niter"]):
                    # generate batch data
                    x_batch, y_batch, aux_batch, mean_batch, std_batch = \
                        load_train_data(cfg, x, y, aux, scaler)
                    print(y_batch)
                    a = y_batch*std_batch+mean_batch
                    soil_depth = [70, 210, 720, 1864.6]  # mm
                    a[:,:4] = a[:,:4]*soil_depth
                    b = np.nansum(a, axis=-1)
                    print(b)
                    with tf.GradientTape() as tape:
                        # cal MSE loss
                        if cfg["model_name"] in ['hard_multi_tasks', 'multi_tasks']:
                            pred = model(x_batch, aux_batch, mean_batch, std_batch, training=True)
                        else:
                            pred = model(x_batch, training=True)
                        mse_loss = RMSELoss(cfg)(y_batch, pred)
                        MSELoss+=mse_loss
                        # cal physic loss
                        if cfg["model_name"] != 'single_task':
                            phy_loss = MassConserveLoss(cfg, mean_batch, std_batch)(
                                aux_batch, pred)
                            MCLoss += phy_loss
                        # cal all loss
                        if cfg["model_name"] in ['single_task', 'multi_tasks', 'hard_multi_tasks']:
                            loss = mse_loss
                        elif cfg["model_name"] in ["soft_multi_tasks"]:
                            loss = (1-cfg["alpha"])*mse_loss + (cfg["alpha"])*phy_loss
                    # gradient tape
                    grads = tape.gradient(loss, model.trainable_variables)
                    optim.apply_gradients(zip(grads, model.trainable_variables))
                    metric.update_state(y_batch, pred)
                # get multi-task loss
                train_acc = metric.result()
                train_1 = train_acc["SWVL_1"].numpy()
                train_2 = train_acc["SWVL_2"].numpy()
                train_3 = train_acc["SWVL_3"].numpy()
                train_4 = train_acc["SWVL_4"].numpy()
                train_5 = train_acc["ET"].numpy()
                train_6 = train_acc["R"].numpy()
                metric.reset_states()
                t1 = time.time()

                # log
                loss_str = "Epoch {} Train NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} SWVL4 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(
                    epoch, train_1, train_2, train_3, train_4, train_5, train_6, 
                    MSELoss/cfg["niter"], MCLoss/cfg["niter"], t1-t0)
                print(loss_str)
                pbar.set_postfix(loss=train_acc)

                # refresh train if loss equal to NaN
                # Will build fresh model and re-train it
                # until it didn't have NaN loss.
                if np.isnan(MSELoss):
                    break

                # validate
                if valid_split:
                    MC_valid_loss = 0
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1

                        # NOTE: We use larger than 3 years for validate, and
                        #       used grids-mean NSE as valid metrics.
                        #       We cannot use `tf.data.Dataset.from_tensor_slices`
                        #       to transform nd.array to tensor. Because it will
                        #       put all valid data into GPU, which exceed memory.
                        t0 = time.time()
                        for i in range(x_valid.shape[0]):
                            if cfg["model_name"] in ['hard_multi_tasks', 'multi_tasks']:
                                pred = model(x_valid[i], aux_valid[i], mean_valid[i], std_valid[i], training=False)
                            else:
                                pred = model(x_valid[i], training=False)
                            mse_valid_loss = RMSELoss(cfg)(y_valid[i], pred)
                            MSE_valid_loss+=mse_valid_loss
                            valid_metric.update_state(y_valid[i], pred)
                            phy_loss = MassConserveLoss(cfg, mean_valid[i], std_valid[i])(aux_valid[i], pred)

                            MC_valid_loss+=phy_loss
                        val_acc = valid_metric.result()
                        val_1 = val_acc["SWVL_1"].numpy()
                        val_2 = val_acc["SWVL_2"].numpy()
                        val_3 = val_acc["SWVL_3"].numpy()
                        val_4 = val_acc["SWVL_4"].numpy()
                        val_5 = val_acc["ET"].numpy()
                        val_6 = val_acc["R"].numpy()

                        valid_metric.reset_states()
                        t1 = time.time()

                        # log in `red`
                        loss_str = '\033[1;31m%s\033[0m' % \
                            "Epoch {} Val NSE SWVL1 {:.3f} SWVL2 {:.3f} SWVL3 {:.3f} SWVL4 {:.3f} ET {:.3f} R {:.3f} MSE Loss {:.3f} MC Loss {:.3f} time {:.2f}".format(
                                epoch, val_1, val_2, val_3, val_4, val_5, val_6, 
                                MSE_valid_loss/x_valid.shape[0], MC_valid_loss/x_valid.shape[0], t1-t0)
                        print(loss_str)

                        # save best model by val loss
                        if MSE_valid_loss < best:
                            model.save_weights(save_folder)
                            wait = 0  # release wait
                            best = MSE_valid_loss
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
