import os
import json

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np
import wandb

from callback import CallBacks
from config import parse_args
from data import  Dataset
from data_generator import DataGenerator
from loss import PHydroLoss
from model import VanillaLSTM, MTLLSTM, MTLHardLSTM
from utils import init_fold
from train import train, predict



def main(cfg):
    print("[PHydro] Now we train {} models".format(cfg["model_name"]))
    
    # init fold for work path
    init_fold(cfg["work_path"])

    # logging params in wandb
    default = dict(# model
                   model_name=cfg["model_name"],
                   n_filter_factors=cfg["n_filter_factors"],
                   # train
                   batch_size=cfg["batch_size"],
                   epochs=cfg["epochs"],
                   alpha=cfg["alpha"])
    wandb.init("PHydro", config=default, allow_val_change=True)

    # load train/valid/test data
    print("[PHydro] Loading train/valid/test datasets")
    f = Dataset(cfg, mode='train')
    x_train, y_train = f.fit()
    f = Dataset(cfg, mode='test')
    x_test, y_test = f.fit()
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)
    os.system("mv {} {}".format("*npy", cfg["inputs_path"]))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # load scaler for inverse
    with open(cfg["inputs_path"] + 'scaler.json', "r") as j:
        scaler = json.load(j)

    # train & inference
    print("[PHydro] Start train!")
    if cfg["model_name"] == 'single_task':
        for i in range(cfg["num_out"]):
            model = VanillaLSTM(cfg)
            train(model, x_train, y_train[:,:,i:i+1], cfg, i=i)
        
        ## pred
        y_pred = predict(x_test, y_test, scaler, cfg)
    else:
        pass

    # save
    print('[PHydro] Saving')
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    np.save(cfg["model_name"]+'_guangdong_9km.npy', y_pred)
    os.system('mv {} {}'.format(cfg["model_name"]+'_guangdong_9km.npy', path))
    os.system('mv {} {}'.format("loss_"+cfg["model_name"]+'.npy', path))


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
