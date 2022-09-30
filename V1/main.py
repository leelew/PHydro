import time
import datetime
from train import train_step
from src.data import Dataset
from src.metrics import RMetrics, MTLRSquareMetrics
from src.model import MTLHydro, LSTMModel, MTLLSTM
from src.loss import MSELoss
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.losses import MeanSquaredError
import logging
from pathlib import Path, PosixPath
import argparse
from tqdm import trange
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def get_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="/work/lilu/MTLHydro/input/")
    parser.add_argument('--cfg_root', type=str,
                        default="/work/lilu/MTLHydro/input/")
    parser.add_argument('--run_dir', type=str,
                        default="/work/lilu/MTLHydro/run/")
    parser.add_argument('--scaler_type', type=str,
                        choices=['global', 'region'])
    parser.add_argument('--scaler_method', type=str, default='minmax')
    parser.add_argument('--is_multivars', type=int, default=-1)
    parser.add_argument('--sample_type', type=str, default='shen')
    parser.add_argument('--num_in', type=int, default=6)
    parser.add_argument('--num_out', type=int, choices=[1, 6])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seq_length', type=int, default=365)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--is_finetune', type=bool, default=False)
    cfg = vars(parser.parse_args())

    # convert path to PosixPath object
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["cfg_root"] = Path(cfg["cfg_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def main(cfg):
    # start logging
    now = time.strftime("%Y-%m-%d_%H:%M:%S")
    logging.basicConfig(
        filename=cfg["run_dir"] / "log" / "run_{date}.log".format(date=now),
        level=logging.DEBUG)
    logging.info("Start MTLHydro")
    logging.info("author: Lu Li")
    logging.info("mail: lilu83@mail.sysu.edu.cn")

    # TODO:logging cfg
    logging.info("config")

    idx = cfg["is_multivars"]

    # load and fit data
    cls_train = Dataset(cfg, is_train='train', interval=7)
    x_train, y_train = cls_train.fit()
    print(x_train.shape, y_train.shape)
    cls_valid = Dataset(cfg, is_train='valid', interval=1)
    x_valid, y_valid = cls_valid.fit()
    print(x_valid.shape, y_valid.shape)
    x_valid, y_valid = cls_valid.fit_transform(x_valid, y_valid)

    if idx != -1:
        y_train, y_valid = y_train[:, :, idx:idx+1], y_valid[:, :, idx:idx+1]
    else:
        pass

    val_data = tf.data.Dataset.from_tensor_slices((
        x_valid.reshape(-1, cfg["seq_length"], cfg["num_in"]),
        y_valid.reshape(-1, cfg["num_out"])))
    val_data = val_data.batch(cfg["batch_size"])

    # FIXME:assert and logging
    assert(np.isnan(x_train).any() == 0 & np.isnan(y_train).any() == 0)
    print(np.nanmin(x_train), np.nanmax(x_train),
          np.nanmin(y_train), np.nanmax(y_train))

    # training setting
    optimizer = Adam(cfg["learning_rate"])
    # MTLRSquareMetrics()#tfa.metrics.RSquare()#RMetrics()
    train_acc_metric = tfa.metrics.RSquare()
    # MTLRSquareMetrics()#tfa.metrics.RSquare()#RMetrics()
    valid_acc_metric = tfa.metrics.RSquare()
    loss_fn = tf.keras.losses.MeanSquaredError()  # MSELoss()
    best_acc, val_acc = -9999, 0

    # get model
    model = MTLLSTM(cfg)  # LSTMModel(cfg)

    def get_variables(trainable_variables, name):
        return [v for v in trainable_variables if name in v.name]

    # fine tune of head
    if cfg["is_finetune"]:
        print(cfg["seq_length"], cfg["num_in"])
        model.build(input_shape=(cfg["batch_size"],
                                 cfg["seq_length"], cfg["num_in"]))
        model.load_weights(cfg["run_dir"] / 'saved_model' /
                           'model_{var}.h5'.format(var=cfg["is_multivars"]))
        for epoch in range(1, cfg["epochs"]+1):
            for iter in range(cfg["niters"]):
                x_batch, y_batch = cls_train.fit_transform(x_train, y_train)
                with tf.GradientTape(persistent=True) as tape:
                    pred = model(x_batch)
                    loss = []
                    for i in range(cfg["num_out"]):
                        loss.append(loss_fn(y_batch[:, i], pred[:, i]))

                trainable_vars = model.trainable_variables
                for i in range(cfg["num_out"]):
                    vars = get_variables(trainable_vars, "head_"+str(i+1))
                    grads = tape.gradient(loss[i], vars)
                    optimizer.apply_gradients(zip(grads, vars))

                train_acc_metric.update_state(y_batch, pred)
            # display metrics at the end of each epoch.
            train_acc = train_acc_metric.result().numpy()
            train_acc_metric.reset_states()

            # run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_data:
                val_logits = model(x_batch_val)
                valid_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = valid_acc_metric.result().numpy()
            valid_acc_metric.reset_states()

            # logging info for each epoch
            logging.info("epoch: {epoch}: train: {train_acc}, valid: {valid_acc}".format(
                epoch=epoch, train_acc=round(train_acc, 2), valid_acc=round(val_acc, 2)))
            print("{epoch}: train: {train_acc}, valid: {valid_acc}".format(
                epoch=epoch, train_acc=round(train_acc, 2), valid_acc=round(val_acc, 2)))

            # save best model
            if val_acc > best_acc:
                model.save_weights(
                    cfg["run_dir"] / 'saved_model' / 'model_ft.h5')
                best_acc = val_acc
    """
    # train
    for epoch in range(1, cfg["epochs"]+1):
        for iter in range(cfg["niters"]):
            x_batch, y_batch = cls_train.fit_transform(x_train, y_train)
            with tf.GradientTape() as tape:
                pred = model(x_batch)
                loss = loss_fn(y_batch, pred)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch, pred)
            
            # Log every 200 iteration.
            #if step % 200 == 0:
            #    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss)))
            #    print("Seen so far: %d samples" % ((step + 1) * ngrids))

        # display metrics at the end of each epoch.
        train_acc = train_acc_metric.result().numpy()
        train_acc_metric.reset_states()

        # run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_data:
            val_logits = model(x_batch_val)
            valid_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = valid_acc_metric.result().numpy()
        valid_acc_metric.reset_states()

        # logging info for each epoch
        logging.info("epoch: {epoch}: train: {train_acc}, valid: {valid_acc}".format(
            epoch=epoch, train_acc=round(train_acc,2), valid_acc=round(val_acc,2)))
        print("{epoch}: train: {train_acc}, valid: {valid_acc}".format(
            epoch=epoch, train_acc=round(train_acc,2), valid_acc=round(val_acc,2)))

        # save best model
        if val_acc > best_acc:
            model.save_weights(cfg["run_dir"] / 'saved_model' / 'model_{var}.h5'.format(var=cfg["is_multivars"]))
            best_acc = val_acc
    """
    # inference TODO:seperate from train mode
    cls_test = Dataset(cfg, is_train='test', interval=1)
    x_test, y_test = cls_test.fit()
    x_test, y_test = cls_test.fit_transform(x_test, y_test)

    if cfg["is_multivars"] == -1:
        y_pred = []
        for x_test_grid in x_test:
            y_pred.append(model.predict(x_test_grid))
        y_pred = np.stack(y_pred, axis=0)
        y_pred = cls_test.reverse_normalize(
            y_pred, "output", cfg["scaler_method"], cfg["is_multivars"])
        y_test = np.stack(y_test, axis=0)
        np.save("y_pred_multivars.npy", y_pred)
        np.save("y_test_multivars.npy", y_test)
    else:
        y_pred = []
        for x_test_grid in x_test:
            y_pred.append(model.predict(x_test_grid))
        y_pred = np.stack(y_pred, axis=0)
        y_pred = cls_test.reverse_normalize(
            y_pred, "output", cfg["scaler_method"], cfg["is_multivars"])
        y_test = np.stack(y_test[:, :, idx:idx+1], axis=0)
        np.save("y_pred_vars_{var}.npy".format(
            var=cfg["is_multivars"]), y_pred)
        np.save("y_test_vars_{var}.npy".format(
            var=cfg["is_multivars"]), y_test)

    logging.info("finished! hope you get a satisfied result!")


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
