from tensorflow.keras.losses import Loss


class PHydroLoss(Loss):
    def __init__(self, cfg):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        return
