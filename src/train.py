import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf
from timm.scheduler import (
    StepLRScheduler,
    MultiStepLRScheduler,
    CosineLRScheduler,
)
from timm.utils.metrics import (
    AverageMeter,
    accuracy,
)

# My library
from dataloader import build_dataloader
from models import build_model

class LightningModel(L.LightningModule):
    def __init__(self,model,cfg):
        super().__init__()
        self.model = model

        self.seq_len = cfg.data.seq_len
        self.num_copies = cfg.data.num_copies
        self.valid_sampling_strategy = cfg.data.sampling_strategy.valid
        self.test_sampling_strategy = cfg.data.sampling_strategy.test

        self.label_smooting = cfg.label_smooting
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smooting)

        self.epochs = cfg.epochs
        self.lr = cfg.optim_args.lr
        self.min_lr = cfg.scheduler_args.min_lr
        self.warmup_epoch = cfg.scheduler_args.warmup_epoch
        self.warmup_lr_init = cfg.scheduler_args.warmup_lr_init

        self.topk = cfg.topk
    
    def forward(self,x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epoch, max_epochs=self.epochs, warmup_start_lr=self.warmup_lr_init, eta_min=self.min_lr,  
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.model(data)
        loss = self.loss_fn(pred,label)

        topk_acc = accuracy(pred, label.argmax(dim=-1) if label.dim()==2 else label, self.topk)

        self.log("train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for top,acc in zip(self.topk, topk_acc):
            self.log(f"train acc(@{top})", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss":loss}
    
    def validation_step(self, batch, batch_idx):
        data, label = batch

        if self.valid_sampling_strategy=="k_copies":
            all_output = []
            stride = data.size()[2] // self.num_copies 
            for j in range(self.num_copies):

                X_slice = data[:, :, j * stride: (j + 1) * stride]
                output = self.model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            pred = torch.mean(all_output, dim=1)
        else:
            pred = self.model(data)

        loss = self.loss_fn(pred, label)
        topk_acc = accuracy(pred, label, self.topk)

        self.log("valid loss",loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for top, acc in zip(self.topk, topk_acc):
            self.log(f"valid acc(@{top})", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        data, label = batch

        if self.valid_sampling_strategy=="k_copies":
            all_output = []
            stride = data.size()[2] // self.num_copies 
            for j in range(self.num_copies):

                X_slice = data[:, :, j * stride: (j + 1) * stride]
                output = self.model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            pred = torch.mean(all_output, dim=1)
        else:
            pred = self.model(data)

        loss = self.loss_fn(pred,label)
        topk_acc = accuracy(pred, label, self.topk)

        self.log("test loss",loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for top, acc in zip(self.topk, topk_acc):
            self.log(f"test acc(@{top})", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@hydra.main(version_base=None, config_path="conf", config_name="default")
def train(cfg : DictConfig) -> None:
    fix_seed(cfg.seed)

    dataloaders = build_dataloader(cfg)
    model = build_model(cfg)

    model = LightningModel(model,cfg)
    logger = TensorBoardLogger(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, name="my_model")
    trainer = L.Trainer(max_epochs=cfg.epochs,accumulate_grad_batches=cfg.accum_iter, precision='bf16-mixed', logger=logger)
    trainer.fit(model=model, train_dataloaders=dataloaders["train"],val_dataloaders=dataloaders["valid"])


if __name__=="__main__":
    train()