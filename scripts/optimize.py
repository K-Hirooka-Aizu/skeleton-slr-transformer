import random
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
from omegaconf import DictConfig, OmegaConf # OmegaConf を追加
from dotenv import load_dotenv
import wandb

# Optuna / Pruning のためのインポート
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# My library
from sstan.datamodule import build_lightning_data_module
from sstan.models import build_model

load_dotenv()


# --- train.py から LightningModel クラスをコピー ---
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
        # self.lr と self.epochs は Optuna によって
        # トライアルごとに異なる値が設定された cfg から渡されます
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epoch, max_epochs=self.epochs, warmup_start_lr=self.warmup_lr_init, eta_min=self.min_lr,  
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        data, label = batch["skeleton_data"], batch["label"]

        pred = self.model(data)
        loss = self.loss_fn(pred,label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        batch_value = self.train_metrics(pred, label.argmax(dim=-1) if label.dim() != 1 else label)
        self.log_dict(batch_value,logger=True,on_epoch=True)

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
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.valid_metrics.update(pred, label.argmax(dim=-1) if label.dim() != 1 else label)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(),logger=True,on_epoch=True)
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
        self.log("test_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_metrics.update(pred, label.argmax(dim=-1) if label.dim() != 1 else label)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(),logger=True,on_epoch=True)
        self.test_metrics.reset()

# --- train.py から fix_seed 関数をコピー ---
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------
# ★★★ ここからがOptuna最適化のためのクラス ★★★
# -----------------------------------------------------------------

class HyperparameterOptimizer:
    """
    Hydra + Optuna を使用してハイパーパラメータ最適化を実行するクラス
    """
    
    def __init__(self, cfg: DictConfig):
        self.base_cfg = cfg
        self.monitor = cfg.optuna.monitor   # 監視メトリクス (例: "valid_accuracy_PI@01")
        self.direction = cfg.optuna.direction # "maximize" or "minimize"
        
        # --- シードを固定 ---
        # DataModuleのシャッフルなどを固定するため、最初に一度実行
        fix_seed(cfg.seed)
        seed_everything(seed=cfg.seed, workers=True)

    def _generate_trial_cfg(self, trial: optuna.trial.Trial) -> DictConfig:
        """
        Optunaのtrialから、このトライアル用のHydra設定(DictConfig)を生成する
        """
        # ベースとなる設定をディープコピー
        trial_cfg = self.base_cfg.copy()

        if not self.base_cfg.get('search_space'):
            return trial_cfg
        
        # search_space に従ってハイパーパラメータを提案し、trial_cfgを上書き
        for key, p in self.base_cfg.search_space.items():

            if p is None:
                continue
            
            if p.type == 'float':
                value = trial.suggest_float(key, p.low, p.high, log=p.get('log', False))
            elif p.type == 'int':
                value = trial.suggest_int(key, p.low, p.high, step=p.get('step', 1))
            elif p.type == 'categorical':
                value = trial.suggest_categorical(key, p.choices)
            else:
                raise ValueError(f"Unknown search_space type: {p.type}")
            
            # OmegaConf.update を使って、ネストしたキー (例: "optim_args.lr") に値を設定
            try:
                OmegaConf.update(trial_cfg, key, value)
            except Exception as e:
                print(f"Warning: Could not update config key '{key}'. Error: {e}")

        return trial_cfg

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Optunaの目的関数 (1トライアル分)
        train.py の main ロジックをここに移植する
        """
        try:
            # 1. このトライアル用の設定(cfg)を生成
            trial_cfg = self._generate_trial_cfg(trial)
            
            # 2. モデルの構築
            # (モデル構造自体を探索する場合、build_modelがtrial_cfgを参照するようにする)
            model_arch = build_model(trial_cfg)
            model = LightningModel(model_arch, trial_cfg)

            datamodule = build_lightning_data_module(self.base_cfg)

            # 3. Loggerの設定
            # ログが混ざらないよう、トライアルごとに保存先を分ける
            wandb_kwargs = OmegaConf.to_container(self.base_cfg.wandb, resolve=True)

            if wandb_kwargs.pop("enabled", False):
                logger = WandbLogger(
                    **wandb_kwargs
                    )
            else:
                logger = TensorBoardLogger(
                    save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    name=f"{self.base_cfg.model.model_name}__{self.base_cfg.data.dataset}"
                )

            trial_cfg_dict = OmegaConf.to_container(trial_cfg, resolve=True)
            logger.log_hyperparams(trial_cfg_dict)

            # 4. Optuna Pruning (枝刈り) Callback
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.monitor)

            # 5. Trainer の設定
            trainer = L.Trainer(
                max_epochs=trial_cfg.epochs, # ★ Optunaで探索したEpoch数
                accumulate_grad_batches=trial_cfg.accum_iter,
                gradient_clip_val=trial_cfg.gradient_clip,
                precision='bf16-mixed', 
                logger=logger,
                callbacks=[
                    pruning_callback,
                    # HPO中はCheckpointを無効化 (速度向上のため)
                    # checkpoint_callback, 
                ],
                deterministic=True,
                enable_checkpointing=False,  # ★ HPO中はFalse推奨
                enable_progress_bar=True,  # ログが綺麗になる
                # (必要に応じて) GPU設定
                # accelerator="gpu", 
                # devices=1,
            )
            
            # 6. 学習実行
            trainer.fit(model=model, datamodule=datamodule)
            
            # 7. スコアを返す
            # Pruning Callbackが監視しているメトリクスを取得
            score = trainer.callback_metrics.get(self.monitor)
            
            if score is None:
                print(f"Warning: Metric '{self.monitor}' not found. Returning worst score.")
                return -1e9 if self.direction == "maximize" else 1e9
            
            logger.experiment.summary[f"best_{self.monitor}"] = score.item()

            return score.item()

        except optuna.exceptions.TrialPruned as e:
            # 枝刈りされた場合
            raise e
        except Exception as e:
            # 学習中のエラー (NaN, OOMなど)
            print(f"Trial {trial.number} failed with error: {e}")
            # エラーの場合は最悪のスコアを返す
            return -1e9 if self.direction == "maximize" else 1e9
        
        finally:
            # ★ 追加: トライアルが終了（正常終了・枝刈り・エラー含む）したら、必ずW&BのRunを終了させる
            wandb.finish()

    def run_optimization(self):
        """
        OptunaのStudyを起動し、最適化を実行する
        """
        # 1. 現在のHydraの出力ディレクトリの絶対パスを取得
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
        # 2. 保存先のDBパスを作成 (sqlite:///{絶対パス}/ファイル名)
        # Linux/Macではスラッシュ4つ(sqlite:////...)、Windowsでは3つになりますが、
        # このf-stringの書き方でOS問わず自動的に正しい形式になります。
        db_url = f"sqlite:///{hydra_output_dir}/optuna_history.db"
        
        print(f"Optuna Database path: {db_url}")

        # Prunerの設定
        pruner = None
        if self.base_cfg.optuna.pruner.enabled:
            pruner = optuna.pruners.MedianPruner(**self.base_cfg.optuna.pruner.pruner_args)
        
        study = optuna.create_study(
            direction=self.direction,
            pruner=pruner,
            storage=db_url,  # ★ YAMLの値ではなく、ここで作ったパスを使う
            study_name=self.base_cfg.optuna.get("study_name", None),
            load_if_exists=True
        )
        
        print(f"Start optimization. Study name: {study.study_name}")
        print(f"Storage: {self.base_cfg.optuna.storage}")

        study.optimize(
            self.objective, 
            n_trials=self.base_cfg.optuna.n_trials,
            timeout=None
        )
        
        print("--- Optimization finished ---")
        print("Best trial:")
        best = study.best_trial
        print(f"  Value (Metric: {self.monitor}): {best.value}")
        print("  Params: ")
        for key, value in best.params.items():
            print(f"    {key}: {value}")

        # --- ★変更点2: CSVデータセットとして保存 ---
        # 全トライアルの結果をPandas DataFrameとして取得
        df = study.trials_dataframe()
        
        # 保存ファイル名 (Hydraの出力ディレクトリに保存されます)
        csv_path = "optuna_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Optimization results saved to: {csv_path}")

        return study

# --- Hydraによるスクリプト実行 ---
# config_path は train.py と同じ場所を指定
@hydra.main(version_base=None, config_path="conf", config_name="default_optimize")
def main(cfg: DictConfig) -> None:
    print("--- Base Configuration (from YAML) ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------------")

    optimizer = HyperparameterOptimizer(cfg)
    optimizer.run_optimization()


if __name__=="__main__":
    main()