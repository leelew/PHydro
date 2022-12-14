"""
Main process for all exps.

<<<<<<<< HEAD
Author: Lu Li
11/1/2022: First edition
"""

import os
import json

import numpy as np
import wandb

from config import parse_args
from data import Dataset
from utils import init_fold
from train import train
from eval import eval_multi, eval_single


def main(cfg):
    print("[PHydro] Train {} model".format(cfg["model_name"]))

    # init fold for work path
    init_fold(cfg["work_path"])

    # logging params in wandb
    print("[PHydro] Wandb info")
    default = dict(  
        # model
        model_name=cfg["model_name"],
        hidden_size=cfg["hidden_size"],
        # train
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        alpha=cfg["alpha"])
    wandb.init("PHydro", config=default, allow_val_change=True)

    # load train/valid/test data
    print("[PHydro] Loading datasets")
    if cfg["reuse_input"]:
        x_train = np.load(cfg["inputs_path"]+'x_train.npy')
        y_train = np.load(cfg["inputs_path"]+'y_train.npy')
        aux_train = np.load(cfg["inputs_path"]+'z_train.npy')
        x_test = np.load(cfg["inputs_path"]+'x_test.npy')
        y_test = np.load(cfg["inputs_path"]+'y_test.npy')
        aux_test = np.load(cfg["inputs_path"]+'z_test.npy')
    else:
        f = Dataset(cfg, mode='train')
        x_train, y_train, aux_train = f.fit()
        f = Dataset(cfg, mode='test')
        x_test, y_test, aux_test = f.fit()
        print(aux_test.shape)
        print(aux_test.shape)
        np.save("x_train.npy", x_train)
        np.save("y_train.npy", y_train)
        np.save("z_train.npy", aux_train)
        np.save("x_test.npy", x_test)
        np.save("y_test.npy", y_test)
        np.save("z_test.npy", aux_test)
        os.system("mv {} {}".format("*npy", cfg["inputs_path"]))
    print('We use {} samples for training'.format(x_train.shape[0]*x_train.shape[1]))
    assert not np.any(np.isnan(x_train))

    # load scaler for inverse
    with open(cfg["inputs_path"] + 'scaler.json', "r") as j:
        scaler = json.load(j)

    # train & inference
    print("[PHydro] Start train!")
    if cfg["model_name"] == 'single_task':
        for i in range(cfg["num_out"]):
            for j in range(cfg["num_repeat"]):
                train(x_train, y_train[:,:,i:i+1], aux_train, scaler, cfg, j, i)
        y_pred = eval_single(x_test, y_test, scaler, cfg)
    else:
        for j in range(cfg["num_repeat"]):
            train(x_train, y_train, aux_train, scaler, cfg, j)
        y_pred = eval_multi(x_test, y_test, aux_test, scaler, cfg)
  
    # save
    print('[PHydro] Saving')
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(cfg["model_name"]+'_gd_9km.npy', y_pred)
    os.system('mv {} {}'.format(cfg["model_name"]+'_gd_9km.npy', path))


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
