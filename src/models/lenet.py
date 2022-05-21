import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODEL_REGISTRY
from .abstract import Base

from torch.nn import CrossEntropyLoss

# ref https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py


@MODEL_REGISTRY.register()
class Lenet(Base):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, y_hat, y, **kwargs):
        return self.loss(y_hat, y.long())

    def init_model(self, NUM_CLASS, IMG_SIZE):
        self.n_classes = NUM_CLASS
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True
        )
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        hidden_size = (IMG_SIZE // 2 - 4) // 2
        hidden_size = hidden_size * hidden_size * 16
        self.fc1 = torch.nn.Linear(
            hidden_size, 120
        )  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(
            120, 84
        )  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(
            84, self.n_classes
        )  # convert matrix with 84 features to a matrix of 10 features (columns)

        self.loss = CrossEntropyLoss()

    def forward(self, x):

        x = F.relu(self.conv1(x))  # (batch_size, 6, IMG_SIZE, IMG_SIZE)

        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)  # (batch_size, 6, IMG_SIZE // 2, IMG_SIZE // 2)

        # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv2(x))  # (batch_size, 16, IMG_SIZE -4, IMG_SIZE -4)

        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)  # (batch_size, 16, IMG_SIZE // 2, IMG_SIZE // 2)

        # first flatten 'max_pool_2_out' to contain batchsize x other columns
        x = x.view(x.shape[0], -1)
        # FC-1, then perform ReLU non-linearity
        x = F.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = F.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        return x

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError


@MODEL_REGISTRY.register()
class LenetDropout(Base):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, y_hat, y, **kwargs):
        return self.loss(y_hat, y.long())

    def init_model(self, NUM_CLASS, IMG_SIZE):
        self.n_classes = NUM_CLASS
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True
        )
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        hidden_size = (IMG_SIZE // 2 - 4) // 2
        hidden_size = hidden_size * hidden_size * 16
        self.fc1 = torch.nn.Linear(
            hidden_size, 120
        )  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(
            120, 84
        )  # convert matrix with 120 features to a matrix of 84 features (columns)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(
            84, self.n_classes
        )  # convert matrix with 84 features to a matrix of 10 features (columns)

        self.loss = CrossEntropyLoss()

    def forward(self, x):

        x = F.relu(self.conv1(x))  # (batch_size, 6, IMG_SIZE, IMG_SIZE)

        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)  # (batch_size, 6, IMG_SIZE // 2, IMG_SIZE // 2)

        # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv2(x))  # (batch_size, 16, IMG_SIZE -4, IMG_SIZE -4)

        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)  # (batch_size, 16, IMG_SIZE // 2, IMG_SIZE // 2)

        # first flatten 'max_pool_2_out' to contain batchsize x other columns
        x = x.view(x.shape[0], -1)
        # FC-1, then perform ReLU non-linearity
        x = F.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = F.relu(self.fc2(x))
        # FC-3

        logits1 = self.fc3(self.dropout1(x))
        logits2 = self.fc3(self.dropout2(x))
        logits3 = self.fc3(self.dropout3(x))
        logits4 = self.fc3(self.dropout4(x))
        logits5 = self.fc3(self.dropout5(x))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        return logits

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
