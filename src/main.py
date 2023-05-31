"""
Main process for all exps.

<<<<<<<< HEAD
Author: Lu Li
11/1/2022: V1.0
"""

import json
import os

import numpy as np
import wandb

from config import parse_args
from data import Dataset
from eval import eval_multi, eval_single
from train import train
from utils import init_fold


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
        x_train, y_train, aux_train, p_train = f.fit()
        f = Dataset(cfg, mode='test')
        x_test, y_test, aux_test, p_test = f.fit()
        np.save("x_train.npy", x_train)
        np.save("y_train.npy", y_train)
        np.save("z_train.npy", aux_train)
        np.save("p_train.npy", p_train)
        np.save("x_test.npy", x_test)
        np.save("y_test.npy", y_test)
        np.save("z_test.npy", aux_test)
        np.save("p_test.npy", p_test)
        os.system("mv {} {}".format("*npy", cfg["inputs_path"]))
    print('We use {} samples for training'.format(x_train.shape[0]*x_train.shape[1]))
    assert not np.any(np.isnan(x_train))
    print(x_train.shape, y_train.shape, aux_train.shape, p_train.shape)
    print(x_test.shape, y_test.shape, aux_test.shape, p_test.shape)

    # spatial CV or temporal CV
    if (cfg['spatial_cv'] != -1) and (cfg['spatial_cv']<4):
        # remove specific grids
        idx = np.arange(cfg["spatial_cv"]*50, cfg["spatial_cv"]*50+50)
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        aux_train = np.delete(aux_train, idx, axis=0)
        print(x_train.shape, y_train.shape, aux_train.shape)

    if cfg['temporal_cv'] == 1:
        # remove summer 173-264
        idx = []
        for i in range(8):
            idx.append(np.arange(365*i+172,365*i+264))
        idx = np.concatenate(idx,axis=0)
        x_train = np.delete(x_train, idx, axis=1)
        y_train = np.delete(y_train, idx, axis=1)
        aux_train = np.delete(aux_train, idx, axis=1)
        print(x_train.shape, y_train.shape, aux_train.shape)

    # load scaler for inverse
    with open(cfg["inputs_path"] + 'scaler.json', "r") as j:
        scaler = json.load(j)

    # train & inference
    print("[PHydro] Start train!")
    if cfg["model_name"] == 'single_task':
        for i in range(cfg["num_out"]):
            for j in range(cfg["num_repeat"]):
                train(x_train, y_train[:,:,i:i+1], aux_train, scaler, cfg, j, i)
        y_pred, y_pred_ens = eval_single(x_test, y_test, scaler, cfg)
    else:
        #for j in range(cfg["num_repeat"]):
        #    train(x_train, y_train, aux_train, scaler, cfg, j)
        y_pred, y_true, aux_true, y_pred_ens = eval_multi(x_test, y_test, aux_test, scaler, cfg)
  
    # save
    print('[PHydro] Saving')
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    if not os.path.exists(path): os.mkdir(path)
    np.save(cfg["model_name"]+'_ens_gd_9km.npy', y_pred_ens)
    np.save(cfg["model_name"]+'_gd_9km.npy', y_pred)
    os.system('mv {} {}'.format(cfg["model_name"]+'_gd_9km.npy', path))
    np.save('obs_gd_9km.npy', y_true)
    np.save('aux_gd_9km.npy', aux_true)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
