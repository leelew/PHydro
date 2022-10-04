import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

from config import get_args
from data import Dataset
from loss import PHydroLoss
from metrics import PHydroMetrics
from model import LSTM, MTLLSTM, MTLHardLSTM, MTLSoftLSTM
from train import (predict_multi_tasks, predict_single_task, train_multi_tasks,
                   train_single_task)


def main(cfg):
    print("[PHydro] Now we train {} models".format(cfg["model_name"]))

    # load train/valid/test data
    f = Dataset(cfg, mode='train')
    x_train, y_train = f.fit()
    f = Dataset(cfg, mode='valid')
    x_valid, y_valid = f.fit()
    f = Dataset(cfg, mode='test')
    x_test, y_test = f.fit()

    # get model/loss
    if cfg["model_name"] == 'single_task':
        model = LSTM()
    elif cfg["model_name"] == 'multi_tasks':
        model = MTLLSTM()
    elif cfg["model_name"] == 'soft_multi_tasks':
        model = MTLSoftLSTM()
    elif cfg["model_name"] == 'hard_multi_tasks':
        model = MTLHardLSTM()

    # get train params
    optim = Adam(cfg["learning_rate"])
    loss_fn = PHydroLoss(cfg)
    train_acc = PHydroMetrics(cfg)
    valid_acc = PHydroMetrics(cfg)
    best_acc, val_acc = -9999, 0  # used to save best model

    # train & inference
    if cfg["model_name"] == 'single_task':
        mdl_list = train_single_task(
            x_train, y_train, cfg, model, loss_fn, optim, f.make_training_data)
        y_pred = predict_single_task(x_test, mdl_list)
    elif cfg["model_name"] == "multi_tasks":
        mdl = train_multi_tasks(
            x_train, y_train, cfg, model, loss_fn, optim, f.make_training_data)
        y_pred = predict_multi_tasks(x_test, mdl)

    # save


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
