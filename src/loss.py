from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error


class PHydroLoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.alpha = cfg["alpha"]  # weight for physical loss (0-1)

    def __call__(self, y_true, y_pred, aux=None):
        if self.model_name in ["single_task", "multi-tasks"]:
            return mean_squared_error(y_true, y_pred)
        elif self.model_name in ["soft_multi_tasks"]:
            # TODO: Cal physical loss
            physical_loss = 0
            return
        elif self.model_name == "hard_multi-tasks":
            return
