import torch
import numpy as np
import logging 

class EarlyStopping:
    """ If the validation loss does not improve after a given amount of patience,
    the training is stopped early.
    """
    def __init__(
            self,
            patience=7,
            verbose=True,
            delta=0,
            save_path='checkpoint.pt'):
        """
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                      Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.delta = -delta
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def reset_all(self):
        self.counter = 0
        # self.best_score = None
        self.early_stop = False

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            logging.info("Validation loss decreased."\
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}) Saving model to {self.save_path}...")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
