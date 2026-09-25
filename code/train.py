import matplotlib
matplotlib.use('Agg')

import csv
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import (
    EXP_NAME, USE_CBAM,
    BUILD_OUT_DIR, TRAIN_LIST, VAL_LIST,
    BATCH_SIZE, EPOCHS, LR,

    AUG_ENABLE, AUG_FLIP_P, AUG_ROT90_P, AUG_NOISE_P, AUG_NOISE_STD,
    AUG_BRIGHTNESS_P, AUG_BRIGHTNESS_SCALE,

    LOSS_MODE,
    BCE_WEIGHT, POS_WEIGHT, NEG_WEIGHT,
    DICE_WEIGHT, FOCAL_WEIGHT, FOCAL_ALPHA, FOCAL_GAMMA,

    USE_WEIGHTED_DICE, WDICE_MODE, WDICE_FG_WEIGHT, WDICE_BG_WEIGHT, WDICE_MAX_FG_WEIGHT,

    get_best_weight_path, get_last_weight_path,

    USE_SCHEDULER, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    EARLY_STOPPING, EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,

    require_run_dir, get_viz_dir, get_train_log_csv, get_train_config_json,
)

from .augment import random_flip_3d, random_rot90_3d, add_noise, brightness_jitter
from .losses import (
    total_loss,
    soft_dice_from_logits,
    dice_loss,
    focal_loss_with_logits,
    weighted_dice_loss,
)
from .utils_io import load_npz, minmax_norm
from .models import UNet3D_Real

logger = logging.getLogger(__name__)

def foreground_penalty_from_logits(logits, min_fg_ratio=0.003):
    """
    防止模型输出接近全0。
    logits: [B,1,D,H,W]
    """
    prob = torch.sigmoid(logits)
    fg_ratio = prob.mean()
    penalty = torch.clamp(min_fg_ratio - fg_ratio, min=0.0)
    return penalty

# =========================================================
# 固定随机种子（完整版本）
# =========================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 工具函数
# =========================================================
def read_list(txt_path: Path):
    if not txt_path.exists():
        raise RuntimeError(f"split file not found: {txt_path}")
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip()]


def safe_filename(name: str) -> str:
    text = str(name)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        text = text.replace(ch, "_")
    return text.strip()


# =========================================================
# Dataset
# =========================================================
class NpzListDataset(Dataset):
    def __init__(self, folder: Path, file_list, is_train=False):
        self.folder = Path(folder)
        self.files = file_list
        self.is_train = is_train

        if len(self.files) == 0:
            raise RuntimeError("empty dataset list")

        missing = [name for name in self.files if not (self.folder / name).exists()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} npz files missing, first: {missing[0]}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        patch, mask = load_npz(self.folder / name)

        patch = minmax_norm(patch).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)

        if self.is_train and AUG_ENABLE:
            rng = np.random.RandomState(SEED + idx)
            patch, mask = random_flip_3d(patch, mask, AUG_FLIP_P, rng=rng)
            patch, mask = random_rot90_3d(patch, mask, AUG_ROT90_P, rng=rng)
            patch = brightness_jitter(patch, AUG_BRIGHTNESS_P, AUG_BRIGHTNESS_SCALE, rng=rng)
            patch = add_noise(patch, AUG_NOISE_P, AUG_NOISE_STD, rng=rng)
            patch = np.clip(patch, 0.0, 1.0)

        x = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return x, y, name


# =========================================================
# 可视化
# =========================================================
def save_viz(epoch, x, y, logits, name):
    patch = x[0, 0].cpu().numpy()
    gt = y[0, 0].cpu().numpy()
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mid = patch.shape[0] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(patch[mid], cmap="gray")
    plt.title("Patch")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(patch[mid], cmap="gray")
    plt.imshow(gt[mid], alpha=0.4, cmap="jet")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(patch[mid], cmap="gray")
    plt.imshow(prob[mid], alpha=0.4, cmap="jet")
    plt.title("Pred")
    plt.axis("off")

    safe_name = safe_filename(name)[:40]
    out = get_viz_dir() / f"epoch_{epoch:03d}_{safe_name}.png"

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================
# Loss
# =========================================================
def compute_loss(logits, y, return_items=False):
    """
    训练主损失：
    1) 优先使用 weighted dice / dice
    2) 可选叠加 focal
    3) 加一个 foreground penalty，防止全0预测
    """
    if USE_WEIGHTED_DICE:
        dice_term = weighted_dice_loss(
            logits,
            y,
            mode=str(WDICE_MODE),
            fg_weight=float(WDICE_FG_WEIGHT),
            bg_weight=float(WDICE_BG_WEIGHT),
            max_fg_weight=float(WDICE_MAX_FG_WEIGHT),
        )
    else:
        dice_term = dice_loss(logits, y)

    focal_term = torch.tensor(0.0, device=logits.device)
    if LOSS_MODE in ("focal", "dice_focal") and float(FOCAL_WEIGHT) > 0:
        focal_term = focal_loss_with_logits(
            logits,
            y,
            alpha=float(FOCAL_ALPHA),
            gamma=float(FOCAL_GAMMA),
        )

    if LOSS_MODE == "dice":
        base_loss = float(DICE_WEIGHT) * dice_term
    elif LOSS_MODE == "focal":
        base_loss = float(FOCAL_WEIGHT) * focal_term
    elif LOSS_MODE == "dice_focal":
        base_loss = float(DICE_WEIGHT) * dice_term + float(FOCAL_WEIGHT) * focal_term
    else:
        raise ValueError(f"Unsupported LOSS_MODE: {LOSS_MODE}")

    fg_penalty = foreground_penalty_from_logits(logits, min_fg_ratio=0.003)
    loss = base_loss + 2.0 * fg_penalty

    if return_items:
        return loss, {
            "dice_term": float(dice_term.detach().item()),
            "focal_term": float(focal_term.detach().item()),
            "fg_penalty": float(fg_penalty.detach().item()),
            "base_loss": float(base_loss.detach().item()),
        }

    return loss


# =========================================================
# 主训练
# =========================================================
def run_train():
    run_dir = require_run_dir()
    train_log_csv = get_train_log_csv()

    with open(train_log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "train_loss", "val_loss", "val_dice"
        ])
    best_path = get_best_weight_path()
    last_path = get_last_weight_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "EXP_NAME": EXP_NAME,
        "USE_CBAM": USE_CBAM,
        "LOSS_MODE": LOSS_MODE,
        "USE_WEIGHTED_DICE": USE_WEIGHTED_DICE,
        "WDICE_MODE": WDICE_MODE,
        "WDICE_FG_WEIGHT": WDICE_FG_WEIGHT,
        "WDICE_BG_WEIGHT": WDICE_BG_WEIGHT,
        "WDICE_MAX_FG_WEIGHT": WDICE_MAX_FG_WEIGHT,
        "DICE_WEIGHT": DICE_WEIGHT,
        "FOCAL_WEIGHT": FOCAL_WEIGHT,
        "FOCAL_ALPHA": FOCAL_ALPHA,
        "FOCAL_GAMMA": FOCAL_GAMMA,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
    }
    get_train_config_json().write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    train_list = read_list(TRAIN_LIST)
    val_list = read_list(VAL_LIST)

    train_ds = NpzListDataset(BUILD_OUT_DIR, train_list, True)
    val_ds = NpzListDataset(BUILD_OUT_DIR, val_list, False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=torch.cuda.is_available(), generator=g)

    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    model = UNet3D_Real(base_ch=16, use_cbam=USE_CBAM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=float(SCHEDULER_FACTOR),
            patience=int(SCHEDULER_PATIENCE),
        )
    best_dice = -1.0
    early_stop = 0

    for ep in range(1, EPOCHS + 1):

        # -------- train --------
        model.train()
        train_loss = 0

        for batch_idx, (x, y, _) in enumerate(tqdm(train_dl)):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)

            if batch_idx == 0:
                loss, loss_items = compute_loss(logits, y, return_items=True)
                logger.info(
                    "Epoch %d debug | USE_WEIGHTED_DICE=%s | LOSS_MODE=%s | "
                    "dice_term=%.6f | focal_term=%.6f | fg_penalty=%.6f | base_loss=%.6f",
                    ep, USE_WEIGHTED_DICE, LOSS_MODE,
                    loss_items["dice_term"],
                    loss_items["focal_term"],
                    loss_items["fg_penalty"],
                    loss_items["base_loss"],
                )
            else:
                loss = compute_loss(logits, y)

            loss.backward()

            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # -------- val --------
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for i, (x, y, name) in enumerate(val_dl):
                x, y = x.to(device), y.to(device)
                logits = model(x)

                loss = compute_loss(logits, y)
                dice = soft_dice_from_logits(logits, y).item()

                val_loss += loss.item()
                val_dice += dice

                if i == 0:
                    save_viz(ep, x, y, logits, name[0])

        val_loss /= len(val_dl)
        val_dice /= len(val_dl)
        if scheduler is not None:
            scheduler.step(val_dice)

        # -------- save --------
        torch.save(model.state_dict(), last_path)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            early_stop = 0
        else:
            early_stop += 1

        logger.info(
            f"Epoch {ep} | train={train_loss:.4f} val={val_loss:.4f} dice={val_dice:.4f}"
        )
        current_lr = opt.param_groups[0]["lr"]

        with open(train_log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep,
                current_lr,
                train_loss,
                val_loss,
                val_dice,
            ])
            
        if EARLY_STOPPING and early_stop >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping")
            break


if __name__ == "__main__":
    try:
        run_train()
    except Exception as e:
        logger.exception("train failed: %s", e)
        raise