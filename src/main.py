import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

from loss import PHydroLoss
from model import LSTM, MTLLSTM, MTLHardLSTM, MTLSoftLSTM
from metrics import PHydroMetrics
from train import train_singletask, train_multitask, predict_singletask, predict_multitask


def main(cfg):
    print("[PHydro] Now we train {} models".format(cfg["model_name"]))

    # load train/test data

    # get model/loss
    if cfg["model_name"] == 'singlevar':
        model = LSTM()
    elif cfg["model_name"] == 'multivars':
        model = MTLLSTM()
    elif cfg["model_name"] == 'soft_multivars':
        model = MTLSoftLSTM()
    elif cfg["model_name"] == 'hard_multivars':
        model = MTLHardLSTM()

    # get train params
    optimizer = Adam(cfg["learning_rate"])
    loss_fn = PHydroLoss(cfg)
    train_acc = PHydroMetrics(cfg)
    valid_acc = PHydroMetrics(cfg)
    best_acc, val_acc = -9999, 0  # used to save best model

    # train & inference
    if cfg["model_name"] == 'singlevar':
        mdls = train_singletask()
        y_pred = predict_singletask(mdls)
    elif cfg["model_name"] == "multivars":
        mdl = train_multitask()
        y_pred = predict_singletask(mdl)

    # save
