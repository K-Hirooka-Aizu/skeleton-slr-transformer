from typing import Union, Any, Callable

from omegaconf import DictConfig

from .transformers import (
    SpatialTemporalTransformer,
    SpatialTemporalTransformerWithClassToken,
)

def build_model(cfg:DictConfig, **kwargs):
    model_name = cfg.model.model_name
    if model_name == "SpatialTemporalTransformer":
        return SpatialTemporalTransformer(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)
    elif model_name == "SpatialTemporalTransformerWithClassToken":
        return SpatialTemporalTransformerWithClassToken(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)
    else:
        raise RuntimeError(f"Model_name [{model_name}] is not implemented.")
