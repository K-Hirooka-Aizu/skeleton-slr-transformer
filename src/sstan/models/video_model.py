from omegaconf import DictConfig
from torchvision.models.video import s3d
from .cnn.pytorch_i3d import InceptionI3d

def build_model(cfg:DictConfig, **kwargs):
    model_name = cfg.model.model_name
    if model_name == "i3d":
        return InceptionI3d(num_classes=cfg.data.num_classes, in_channels=cfg.data.in_channels ,**cfg.model.model_args)
    elif model_name == "s3d":
        return s3d(num_classes=cfg.data.num_classes)
    else:
        raise RuntimeError(f"Model_name [{model_name}] is not implemented.")
