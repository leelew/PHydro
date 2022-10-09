import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
from callback import CallBacks
from config import parse_args
from data import DataGenerator, Dataset
from loss import PHydroLoss
from metrics import PHydroMetrics
from model import VanillaLSTM, MTLLSTM, MTLHardLSTM
import wandb

def main(cfg):
    print("[PHydro] Now we train {} models".format(cfg["model_name"]))
    wandb.init("PHydro")
    # load train/valid/test data
    print("[PHydro] Loading train/valid/test datasets")
    f = Dataset(cfg, mode='train')
    x_train, y_train = f.fit()
    f = Dataset(cfg, mode='valid')
    x_valid, y_valid = f.fit()
    f = Dataset(cfg, mode='test')
    x_test, y_test = f.fit()

    """
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_valid.npy", x_valid)
    np.save("y_valid.npy", y_valid)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)
    os.system("mv {} {}".format("*npy", cfg["inputs_path"]))
    """

    # load scaler for inverse
    with open(cfg["inputs_path"] + 'scaler.json', "r") as j:
        scaler = json.load(j)

    # train & inference
    if cfg["model_name"] == 'single_task':
        for i in range(cfg["num_out"]):
            model = VanillaLSTM(cfg)
            # params
            callback_path = cfg["outputs_path"]+'saved_model/'+cfg["model_name"]+'/'+str(i)+'/'
            # compile model
            model.compile(optimizer=Adam(cfg["learning_rate"]),
                          loss='mse',#PHydroLoss(cfg),
                          metrics=['mse'])  # PHydroMetrics(cfg))
            # generators
            train_generator = DataGenerator(x_train, y_train[:, :, i], cfg)
            valid_generator = DataGenerator(x_valid, y_valid[:, :, i], cfg)
            # fit
            mdl = model.fit_generator(
                train_generator,
                validation_data=valid_generator,
                epochs=cfg["epochs"],
                steps_per_epoch=cfg["niter"],
                callbacks=[CallBacks(callback_path)()])
        # inference
        # (ngrids, samples, seq_len, nfeat)
        y_pred = []
        for i in range(x_test.shape[0]):  # for each grids
            tmp = []
            for mdl in range(mdl_list):  # for each feat
                model = VanillaLSTM(cfg)
                callback_path = cfg["outputs_path"]+'saved_model/'+cfg["model_name"]+'/'+str(i)+'/'
                model.load_weight(callback_path)
                pred = model.predict(x_test[i])
                pred = f.reverse_normalize(pred, scaler)
                tmp.append(pred)
            tmp = np.concatenate(tmp, axis=-1)  # (samples, num_out)
            y_pred.append(tmp)
        y_pred = np.stack(y_pred, axis=0)  # (ngrids, samples, num_out)
    else:
        # get model
        if cfg["model_name"] in ['multi_tasks', 'soft_multi_tasks']:
            model = MTLLSTM(cfg)
        elif cfg["model_name"] == 'hard_multi_tasks':
            model = MTLHardLSTM(cfg)
        # compile model
        model.compile(optimizer=Adam(cfg["learning_rate"]),
                      loss=PHydroLoss(cfg),
                      metrics=['mse'])  # PHydroMetrics(cfg))
        # generators
        train_generator = DataGenerator(x_train, y_train, cfg)
        valid_generator = DataGenerator(x_valid, y_valid, cfg)
        # fit
        mdl = model.fit_generator(
            train_generator,
            validation_data=valid_generator,
            steps_per_epochs=cfg["niter"],
            callback=[CallBacks(callback_path)()])

        # inference
        y_pred = []
        for i in range(x_test.shape[0]):  # for each grids
            tmp = []
            pred = mdl.predict(x_test[i])
            pred = f.reverse_normalize(pred, scaler)
            y_pred.append(pred)  # (samples, num_out)
        y_pred = np.stack(y_pred, axis=0)  # (ngrids, samples, num_out)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
