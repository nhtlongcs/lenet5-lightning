import abc
from typing import Any

import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST, FashionMNIST
from src.metrics import METRIC_REGISTRY
import torchvision
from torch.utils.data import DataLoader, random_split
from src.utils.device import detach
from torch.utils.data import DataLoader

from . import MODEL_REGISTRY
from .dataset import Caltech256

dataset_factory = {
    'mnist': MNIST,
    'fashion_mnist': FashionMNIST,
    'caltech256': Caltech256
}
@MODEL_REGISTRY.register()
class Base(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.data_dir = config["data"]["data_dir"]
        self.dataset_name = config["data"]["name"]
        self.init_model(**config["model"]["args"])
        self.learning_rate = self.cfg.trainer["lr"]

    @abc.abstractmethod
    def init_model(self):
        raise NotImplementedError

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        # # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        # # download
        Caltech256(self.data_dir, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        image_size = self.cfg["data"]["image_size"]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if stage == "fit" or stage is None:
            if self.dataset_name == "caltech256":
                dataset_full = Caltech256(self.data_dir, transform=self.transform)
            else: 
                dataset_full = dataset_factory[self.dataset_name](self.data_dir, train=True, transform=self.transform)
            total_len = len(dataset_full)
            val_len = max(int(total_len * 0.1), 1)
            train_len = total_len - val_len
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [train_len, val_len]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if self.dataset_name == "caltech256":
                self.dataset_test = Caltech256(self.data_dir, transform=self.transform)
            else: 
                self.dataset_test = dataset_factory[self.dataset_name](self.data_dir, train=True, transform=self.transform)            

        self.metric = {
            mcfg["name"]: METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
            for mcfg in self.cfg["metric"]
        }

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, output, targets):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # 1. Get embeddings from model
        output = self.forward(inputs)
        # 2. Calculate loss
        loss = self.compute_loss(output, targets).mean()
        # 3. Update monitor
        self.log("train_loss", detach(loss))

        return {"loss": loss, "log": {"train_loss": detach(loss)}}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # 1. Get embeddings from model
        output = self.forward(inputs)
        # 2. Calculate loss
        loss = self.compute_loss(output, targets).mean()
        # 3. Update metric for each inputs
        for m in self.metric.keys():
            self.metric[m].update(output, targets)

        return {"loss": detach(loss)}

    def validation_epoch_end(self, outputs):

        # 1. Calculate average validation loss
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # 2. Calculate metric value
        out = {"val_loss": loss}

        # 3. Update metric for each batch
        metric_dict = {}
        for m in self.metric.keys():
            metric_dict.update(self.metric[m].value())
            out.update(metric_dict)

        for k in metric_dict.keys():
            self.log(f"val_{k}", out[k])

        # Log string
        log_string = ""
        for metric, score in out.items():
            if isinstance(score, (int, float)):
                log_string += metric + ": " + f"{score:.5f}" + " | "
        log_string += "\n"
        # print(log_string)

        # 4. Reset metric
        for m in self.metric.keys():
            self.metric[m].reset()
        self.log("val_loss", loss.cpu().numpy().item())
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def train_dataloader(self):
        train_loader = DataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset=self.dataset_train,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            **self.cfg["data"]["args"]["val"]["loader"], dataset=self.dataset_val
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            **self.cfg["data"]["args"]["test"]["loader"],
            dataset=self.dataset_test,
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)
        train_set_len = len(self.dataset_train)
        train_bs = self.cfg["data"]["args"]["train"]["loader"]["batch_size"]

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.trainer["lr"],
            steps_per_epoch=int(train_set_len // train_bs),
            epochs=self.cfg.trainer["num_epochs"],
            anneal_strategy="linear",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
