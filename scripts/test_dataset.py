from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from sstan.datamodule import build_lightning_data_module

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg:DictConfig):
    datamodule = build_lightning_data_module(cfg)
    datamodule.setup(None)

    train_dataloader = datamodule.train_dataloader()
    for i, ((video,mask), label) in enumerate(tqdm(train_dataloader)):
        if i % 1000 == 0:
            print(f"{video.shape} {mask.shape}, {label.shape}")
        
        pass
    valid_dataloader = datamodule.val_dataloader()
    for i, ((video,mask), label) in enumerate(tqdm(valid_dataloader)):
        if i % 1000 == 0:
            print(f"{video.shape} {mask.shape}, {label.shape}")
        
    test_dataloader = datamodule.test_dataloader()
    for ((video,mask), label) in tqdm(test_dataloader):
        pass



if __name__=="__main__":
    main()