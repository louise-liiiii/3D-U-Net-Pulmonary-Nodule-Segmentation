from pathlib import Path
from datetime import datetime
import json

# =========================================================
# 项目根目录
# config.py 位于: D:\LungSeg_GradProject\src\config.py
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# 原始数据目录
# =========================================================
RAW_DIR = Path(r"D:\LUNA16")


# =========================================================
# 项目内生成目录
# =========================================================
PROCESSED_DIR  = PROJECT_ROOT / "01_data_processed"
RESULTS_DIR    = PROJECT_ROOT / "02_results"


# =========================================================
# 处理后数据子目录
# =========================================================
NPZ_DIR   = PROCESSED_DIR / "npz"
SPLIT_DIR = PROCESSED_DIR / "splits"


# =========================================================
# 结果子目录
# =========================================================
RUNS_DIR    = RESULTS_DIR / "runs"
REPORTS_DIR = RESULTS_DIR / "reports"

# =========================================================
# 自动创建目录
# =========================================================
for _d in [
    NPZ_DIR,
    SPLIT_DIR,
    RUNS_DIR,
    REPORTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

def validate_raw_dir():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

# =========================================================
# 数据构建
# =========================================================
LUNA_SUBSET0_DIR = RAW_DIR / "subset0" / "subset0"
LUNA_ANN_CSV     = RAW_DIR / "CSVFILES" / "CSVFILES" / "annotations.csv"

PATCH_SIZE       = 64
MIN_DIAMETER_MM  = 3.0
MAX_DIAMETER_MM  = 10.0
TARGET_SPACING_XYZ = (1.0, 1.0, 1.0)

# =========================================================
# 肺分割参数
# =========================================================
LUNG_SEG_THRESHOLD = -450.0

BUILD_OUT_DIR = NPZ_DIR/ f"subset0_{MIN_DIAMETER_MM:g}_{MAX_DIAMETER_MM:g}mm_p{PATCH_SIZE}_sp1.0"
BUILD_OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 数据划分
# =========================================================
TRAIN_LIST = SPLIT_DIR / "train.txt"
VAL_LIST   = SPLIT_DIR / "val.txt"
TEST_LIST  = SPLIT_DIR / "test.txt"

# =========================================================
# 实验开关   poch 100
# 可选：
# - baseline_dice           poch 28 15min-
# - baseline_dice_focal     poch 23 15min-
# - ours_cbam_dice          poch 46 30min+
# - ours_cbam_dice_focal    poch 39 30min+
# =========================================================

EXPERIMENT = "ours_cbam_dice"

if EXPERIMENT == "baseline_dice":
    USE_CBAM = False
    LOSS_MODE = "dice"
    USE_WEIGHTED_DICE = True
elif EXPERIMENT == "baseline_dice_focal":
    USE_CBAM = False
    LOSS_MODE = "dice_focal"
    USE_WEIGHTED_DICE = True
elif EXPERIMENT == "ours_cbam_dice":
    USE_CBAM = True
    LOSS_MODE = "dice"
    USE_WEIGHTED_DICE = True
elif EXPERIMENT == "ours_cbam_dice_focal":
    USE_CBAM = True
    LOSS_MODE = "dice_focal"
    USE_WEIGHTED_DICE = True
else:
    raise ValueError(f"Unsupported EXPERIMENT: {EXPERIMENT}")

EXP_NAME = EXPERIMENT 

# =========================================================
# Run info（统一由 main.py调用）
# =========================================================
RUN_ID = None
RUN_DIR = None

LAST_WEIGHT_PATH = None   # 占位，待 RUN_DIR 设置后更新
def set_run_dir(run_dir: Path):
    global RUN_ID, RUN_DIR, LAST_WEIGHT_PATH
    RUN_DIR = Path(run_dir)
    RUN_ID = RUN_DIR.name
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # 重新计算 LAST_WEIGHT_PATH
    LAST_WEIGHT_PATH = get_last_weight_path()

def init_new_run():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / EXP_NAME / run_id
    set_run_dir(run_dir)
    return RUN_DIR

def require_run_dir() -> Path:
    if RUN_DIR is None:
        raise RuntimeError("RUN_DIR 未初始化，请先通过 main.py 选择或创建 run")
    return RUN_DIR

def get_weights_dir() -> Path:
    d = require_run_dir() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_best_weight_path() -> Path:
    return get_weights_dir() / "best.pth"

def get_last_weight_path() -> Path:
    return get_weights_dir() / "last.pth"


# =========================================================
# 训练参数
# =========================================================
BATCH_SIZE = 1
EPOCHS     = 100
LR         = 0.001

def get_train_log_csv() -> Path:
    return require_run_dir() / "train_log.csv"

def get_train_config_json() -> Path:
    return require_run_dir() / "train_config.json"

# =========================================================
# 推理参数
# =========================================================
DEFAULT_THRESHOLD = 0.40

def get_best_threshold_json() -> Path:
    return get_metrics_dir() / "best_threshold.json"

def save_best_threshold(thr: float):
    path = get_best_threshold_json()
    path.write_text(
        json.dumps({"best_threshold": float(thr)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def load_best_threshold(default: float = None) -> float:
    if default is None:
        default = DEFAULT_THRESHOLD
    path = get_best_threshold_json()
    if not path.exists():
        return float(default)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return float(obj.get("best_threshold", default))
    except Exception:
        return float(default)

# =========================================================
# Connected Component 后处理
# =========================================================
POSTPROCESS_ENABLE = True

# 可选:
# - "none"
# - "largest"
# - "center"
# - "remove_small"
# - "center_and_small"
POSTPROCESS_METHOD = "center_and_small"

POSTPROCESS_CONNECTIVITY = 26
POSTPROCESS_MIN_SIZE = 30


# =========================================================
# run 输出目录
# =========================================================
def get_viz_dir() -> Path:
    d = require_run_dir() / "viz"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_viz_ct_dir() -> Path:
    d = require_run_dir() / "viz_ct"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_metrics_dir() -> Path:
    d = require_run_dir() / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_eval_csv() -> Path:
    return get_metrics_dir() / "eval.csv"

def get_eval_summary_txt() -> Path:
    return get_metrics_dir() / "eval_summary.txt"

def get_gui_outputs_dir() -> Path:
    d = RESULTS_DIR / "gui_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d

# =========================================================
# 报告输出
# =========================================================
def get_report_dir() -> Path:
    d = REPORTS_DIR / (RUN_ID or "no_run")
    d.mkdir(parents=True, exist_ok=True)
    return d

# =========================================================
# 数据增强
# =========================================================
AUG_ENABLE            = True
AUG_FLIP_P            = 0.5
AUG_ROT90_P           = 0.5
AUG_NOISE_P           = 0.3
AUG_NOISE_STD         = 0.03
AUG_BRIGHTNESS_P      = 0.3
AUG_BRIGHTNESS_SCALE  = 0.15

# =========================================================
# Loss 参数
# =========================================================
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

if LOSS_MODE == "dice":
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 0.0
elif LOSS_MODE == "focal":
    DICE_WEIGHT = 0.0
    FOCAL_WEIGHT = 0.3
elif LOSS_MODE == "dice_focal":
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 0.3
else:
    raise ValueError(f"Unsupported LOSS_MODE: {LOSS_MODE}")


BCE_WEIGHT = FOCAL_WEIGHT
POS_WEIGHT = FOCAL_ALPHA
NEG_WEIGHT = 1.0

# =========================================================
# Weighted Dice 参数
# =========================================================
# USE_WEIGHTED_DICE 由上面的 EXPERIMENT 自动决定

# 可选: "fixed" 或 "dynamic"
WDICE_MODE = "fixed"

# fixed 模式下使用
WDICE_FG_WEIGHT = 5.0
WDICE_BG_WEIGHT = 1.0

# dynamic 模式下用于限制最大前景权重
WDICE_MAX_FG_WEIGHT = 20.0


# =========================================================
# 训练策略
# =========================================================
USE_SCHEDULER        = True
SCHEDULER_PATIENCE   = 3
SCHEDULER_FACTOR     = 0.5

EARLY_STOPPING          = True
EARLY_STOPPING_PATIENCE = 6

GRAD_CLIP_NORM = 1.0