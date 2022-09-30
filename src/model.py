import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K


class LSTM(Model):
    """LSTM with single outputs"""
    pass


class MTLLSTM(Model):
    """LSTM with multivars outputs."""
    pass


class MTLSoftLSTM(Model):
    """LSTM with soft physical constrain through loss"""
    pass


class MTLHardLSTM(Model):
    """LSTM with hard physical constrain through residual layer"""
    pass
