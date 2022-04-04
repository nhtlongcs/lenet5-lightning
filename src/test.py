import torch
from opt import Opts

import pytorch_lightning as pl

from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.callbacks import RichProgressBar

from src.models import MODEL_REGISTRY
import os


def train(config):
    model = MODEL_REGISTRY.get(config["model"]["name"])(config)
    pretrained_path = config["global"]["pretrained"]
    assert os.path.exists(pretrained_path), "Pretrained model not found"
    model = model.load_from_checkpoint(pretrained_path).eval()

    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16 if cfg["global"]["use_fp16"] else 32,
        callbacks=RichProgressBar(),
    )
    trainer.test(model)


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    train(cfg)
