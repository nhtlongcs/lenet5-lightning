from typing import List
import numpy as np
import torch

from registry import Registry

METRIC_REGISTRY = Registry("METRIC")


@METRIC_REGISTRY.register()
class Accuracy:
    """
    Accuracy metric
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        y = y.cpu().numpy()

        y_hat = y_hat.cpu().numpy()
        y_hat = np.argmax(y_hat, axis=-1)

        res = y_hat == y
        self.correct += np.sum(res, axis=-1)
        self.sample_size += res.shape[0]

    def reset(self):
        self.score = 0
        self.correct = 0
        self.sample_size = 0

    def value(self):
        self.score = self.correct / self.sample_size * 100
        return {"Accuracy": self.score}

