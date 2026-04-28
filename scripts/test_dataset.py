from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from sstan.datamodule import build_lightning_data_module

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg:DictConfig):
    datamodule = build_lightning_data_module(cfg)
    datamodule.setup(None)

    train_dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(tqdm(train_dataloader)):
        if i % 1000 == 0:
            print(f"{batch['pixel_values'].shape} {batch['label'].shape}")
        pass
    valid_dataloader = datamodule.val_dataloader()
    for i, batch in enumerate(tqdm(valid_dataloader)):
        if i % 1000 == 0:
            print(f"{batch['pixel_values'].shape} {batch['label'].shape}")
        
    test_dataloader = datamodule.test_dataloader()
    for batch in tqdm(test_dataloader):
        pass



if __name__=="__main__":
    main()