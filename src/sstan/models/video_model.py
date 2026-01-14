from omegaconf import DictConfig
import torch.nn as nn
from torchvision.models.video import s3d, mvit_v2_s, r3d_18

from .cnn.pytorch_i3d import InceptionI3d

def build_model(cfg:DictConfig, **kwargs):
    model_name = cfg.model.model_name
    if model_name == "i3d":
        return InceptionI3d(num_classes=cfg.data.num_classes, in_channels=cfg.data.in_channels ,**cfg.model.model_args)
    elif model_name == "s3d":
        if cfg.model.model_args.pretrained:
            s3d_model = s3d(weights='KINETICS400_V1')
            s3d_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv3d(1024, cfg.data.num_classes, kernel_size=1, stride=1, bias=True),
            )
        else:
            s3d_model = s3d(num_classes=cfg.data.num_classes)
        return s3d_model
    
    elif model_name == "r3d_18":
        if cfg.model.model_args.pretrained:
            r3d_model = r3d_18(weights='KINETICS400_V1')
            _in_features = r3d_model.fc.in_features
            r3d_model.fc = nn.Linear(_in_features, cfg.data.num_classes)
        else:
            r3d_model = r3d_18(num_classes=cfg.data.num_classes)
        return r3d_model
    
    else:
        raise RuntimeError(f"Model_name [{model_name}] is not implemented.")
