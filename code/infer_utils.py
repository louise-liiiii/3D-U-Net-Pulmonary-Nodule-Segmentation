from pathlib import Path

import numpy as np
import torch

from .config import (
    USE_CBAM,
    POSTPROCESS_ENABLE,
    POSTPROCESS_METHOD,
    POSTPROCESS_CONNECTIVITY,
    POSTPROCESS_MIN_SIZE,
    load_best_threshold,
    get_best_weight_path,
)
from .models import UNet3D_Real
from .postprocess import postprocess_nodule_mask


def build_model(device):
    model = UNet3D_Real(base_ch=16, use_cbam=USE_CBAM).to(device)
    return model


def load_model_for_inference(device, weight_path: Path = None):
    if weight_path is None:
        weight_path = get_best_weight_path()

    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise RuntimeError(f"weight not found: {weight_path}")

    model = build_model(device)
    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_prob_from_patch(model, patch_3d: np.ndarray, device):
    if patch_3d.ndim != 3:
        raise ValueError(f"patch_3d must be 3D, got shape={patch_3d.shape}")
    if not np.isfinite(patch_3d).all():
        raise ValueError("patch_3d contains NaN or Inf")

    x = torch.from_numpy(patch_3d.astype(np.float32))[None, None].to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy().astype(np.float32)

    return prob


def prob_to_mask(prob: np.ndarray, threshold: float = None, center=None):
    """
    将概率图转换为二值掩膜，可选后处理。
    center: 结节中心在 patch 中的坐标 (z,y,x)，用于 center 模式的后处理。
           若为 None，则使用几何中心。
    """
    if threshold is None:
        threshold = load_best_threshold()

    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    pred = (prob > threshold).astype(np.uint8)

    if POSTPROCESS_ENABLE:
        if center is None:
            center = tuple(s // 2 for s in pred.shape)
        else:
            center = tuple(int(v) for v in center)
            if len(center) != 3:
                raise ValueError(f"center must have length 3, got {center}")
            if not all(0 <= c < s for c, s in zip(center, pred.shape)):
                raise ValueError(f"center out of bounds: center={center}, shape={pred.shape}")

        pred = postprocess_nodule_mask(
            pred,
            method=POSTPROCESS_METHOD,
            connectivity=POSTPROCESS_CONNECTIVITY,
            min_size=POSTPROCESS_MIN_SIZE,
            center=center,
        )
    return pred


def predict_mask_from_patch(model, patch_3d: np.ndarray, device, threshold: float = None, center=None):
    """
    返回 (概率图, 二值掩膜)
    center: 结节中心坐标，用于后处理，同 prob_to_mask
    """
    prob = predict_prob_from_patch(model, patch_3d, device)
    pred = prob_to_mask(prob, threshold=threshold, center=center)
    return prob, pred