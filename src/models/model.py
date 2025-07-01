from typing import Union, Any, Callable

from omegaconf import DictConfig

from .transformers import (
    SpatialTemporalTransformer,
    SpatialTemporalTransformerWithClassToken,
    PreNormSpatialTemporalTransformer,
    PreNormSpatialTemporalTransformerWithClassToken,
)
from .gcn import(
    STGCN,
    CTRGCN
)

def build_model(cfg:DictConfig, **kwargs):
    model_name = cfg.model.model_name
    if model_name == "SpatialTemporalTransformer":
        return SpatialTemporalTransformer(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)
    elif model_name == "SpatialTemporalTransformerWithClassToken":
        return SpatialTemporalTransformerWithClassToken(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)

    elif model_name == "PreNormSpatialTemporalTransformer":
        return PreNormSpatialTemporalTransformer(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)
    elif model_name == "PreNormSpatialTemporalTransformerWithClassToken":
        return PreNormSpatialTemporalTransformerWithClassToken(in_channels=cfg.data.in_channels, num_classes=cfg.data.num_classes, seq_len=cfg.data.seq_len, n_joints=cfg.data.n_joints, **cfg.model.model_args, **kwargs)

    elif model_name == "stgcn":
        return STGCN(in_channels=cfg.data.in_channels, num_class=cfg.data.num_classes, graph_args=cfg.model.graph_args, edge_importance_weighting=cfg.model.edge_importance_weighting)
    elif model_name == "ctrgcn":
        return CTRGCN(num_class=cfg.data.num_classes, num_point=cfg.data.n_joints, num_person=1, graph_args=cfg.model.graph_args, in_channels=cfg.data.in_channels, drop_out=cfg.model.drop_out, adaptive=cfg.model.adaptive)
    else:
        raise RuntimeError(f"Model_name [{model_name}] is not implemented.")
