import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch import seed_everything
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import wandb
from ptflops import get_model_complexity_info

# My library
from sstan.datamodule import build_lightning_data_module
from sstan.models import build_model

load_dotenv()

class LightningModel(L.LightningModule):
    def __init__(self,model:nn.Module,cfg:DictConfig):
        super().__init__()
        self.model = model

        self.seq_len = cfg.data.seq_len
        self.num_copies = cfg.data.num_copies
        try:
            self.valid_sampling_strategy = cfg.data.sampling_strategy.valid
            self.test_sampling_strategy = cfg.data.sampling_strategy.test
        except:
            self.valid_sampling_strategy = None
            self.test_sampling_strategy = None

        self.label_smooting = cfg.label_smooting
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smooting)

        self.epochs = cfg.epochs
        self.lr = cfg.optim_args.lr
        self.min_lr = cfg.scheduler_args.min_lr
        self.warmup_epoch = cfg.scheduler_args.warmup_epoch
        self.warmup_lr_init = cfg.scheduler_args.warmup_lr_init

        self.topk = sorted(cfg.topk)

        self.train_metrics = torchmetrics.MetricCollection(
            {
                f"accuracy_PI@{str(k).zfill(2)}": torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.data.num_classes, average="micro", top_k=k) for k in self.topk
            }|{
                f"accuracy_PC@{str(k).zfill(2)}": torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.data.num_classes, average="macro", top_k=k) for k in self.topk
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
    
    def forward(self,x):
        return self.model(x)

    def configure_optimizers(self):
        num_gpus = self.trainer.world_size
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr*num_gpus)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=self.warmup_epoch, 
            max_epochs=self.epochs, 
            warmup_start_lr=self.warmup_lr_init * num_gpus,  
            eta_min=self.min_lr * num_gpus
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        data, label = batch["skeleton_data"], batch["label"]

        pred = self.model(data)
        loss = self.loss_fn(pred,label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        batch_value = self.train_metrics(pred, label.argmax(dim=-1) if label.dim() != 1 else label)
        self.log_dict(batch_value,logger=True,on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        data, label = batch["skeleton_data"], batch["label"]

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
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        self.valid_metrics.update(pred, label.argmax(dim=-1) if label.dim() != 1 else label)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(),logger=True,on_epoch=True,sync_dist=True)
        self.valid_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        data, label = batch["skeleton_data"], batch["label"]

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
        self.log("test_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        self.test_metrics.update(pred, label.argmax(dim=-1) if label.dim() != 1 else label)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(),logger=True,on_epoch=True,sync_dist=True)
        self.test_metrics.reset()

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
    seed_everything(seed=cfg.seed, workers=True)
    
    model = build_model(cfg)

    model.eval()

    # 3. 入力データのShapeを定義 (バッチサイズを含まない次元を指定)
    # 【重要】ここはご自身のスケルトンデータに合わせて変更してください！
    # 例: (チャンネル数, フレーム数, 関節数) = (3, 100, 25) の場合
    input_shape = (cfg.data.in_channels, cfg.data.seq_len, cfg.data.n_joints, 1) 

    # 4. FLOPsとパラメータ数の計算
    macs, params = get_model_complexity_info(
        model, 
        input_shape, 
        as_strings=True,            # 見やすい単位（GMac, Mなど）の文字列で出力
        print_per_layer_stat=True,  # 層ごとの計算量を出力するかどうか
        verbose=True
    )

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    

if __name__=="__main__":
    train()

# 3.69 GMac
# 7.38 GMac
# 11.07 GMac
# 14.77 GMac
# 18.46 GMac

# 3.94 GMac
# 8.32 GMac
# 18.46 GMac
# 44.24 GMac