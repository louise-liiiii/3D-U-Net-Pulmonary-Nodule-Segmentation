import sys
from pathlib import Path
import logging

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# 导入项目配置模块
import src.config as config

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QSpinBox, QFormLayout, QComboBox, QDoubleSpinBox,
    QSizePolicy, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor, QBrush, QPen, QPolygon, QFont

from src.models import UNet3D_Real
from src.lung_seg import segment_lung_mask
from src.utils_io import minmax_norm
from src.infer_utils import predict_mask_from_patch
from src.geom_utils import safe_crop_3d, make_sphere_mask
from src.postprocess import postprocess_nodule_mask

# 使用 config 中的常量
SAVE_DIR = config.get_gui_outputs_dir()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WINDOW_LEVEL = -600.0
DEFAULT_WINDOW_WIDTH = 1500.0
DEFAULT_LUNG_ALPHA = 0.18
DEFAULT_NODULE_ALPHA = 0.40
DEFAULT_PATCH_GT_ALPHA = 0.35
DEFAULT_PATCH_PRED_ALPHA = 0.45


def hu_window_to_uint8(img2d: np.ndarray, wl: float, ww: float) -> np.ndarray:
    arr = img2d.astype(np.float32)
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / max(vmax - vmin, 1e-6)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def safe_filename(name: str) -> str:
    text = str(name)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        text = text.replace(ch, "_")
    return text.strip()


def dice_score_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = (pred * gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))

def paste_mask_back_3d(canvas, patch_mask, cz, cy, cx):
    patch = patch_mask.shape[0]
    half = patch // 2
    z1, z2 = cz - half, cz + half
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    Z, Y, X = canvas.shape

    src_z1 = 0
    src_z2 = patch
    src_y1 = 0
    src_y2 = patch
    src_x1 = 0
    src_x2 = patch

    if z1 < 0:
        src_z1 = -z1
        z1 = 0
    if y1 < 0:
        src_y1 = -y1
        y1 = 0
    if x1 < 0:
        src_x1 = -x1
        x1 = 0

    if z2 > Z:
        src_z2 = patch - (z2 - Z)
        z2 = Z
    if y2 > Y:
        src_y2 = patch - (y2 - Y)
        y2 = Y
    if x2 > X:
        src_x2 = patch - (x2 - X)
        x2 = X

    canvas_slice = canvas[z1:z2, y1:y2, x1:x2]
    patch_slice = patch_mask[
        src_z1:src_z2,
        src_y1:src_y2,
        src_x1:src_x2
    ].astype(canvas.dtype)

    # 取最大值，避免覆盖
    canvas[z1:z2, y1:y2, x1:x2] = np.maximum(canvas_slice, patch_slice)
    return canvas


def overlay_mask_on_hu(gray_hu: np.ndarray, mask01: np.ndarray, color="red", alpha=0.45,
                       wl: float = DEFAULT_WINDOW_LEVEL, ww: float = DEFAULT_WINDOW_WIDTH) -> np.ndarray:
    base = hu_window_to_uint8(gray_hu, wl=wl, ww=ww)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    mask_bool = mask01 > 0

    if color == "red":
        rgb[mask_bool, 0] = 255
        rgb[mask_bool, 1] = rgb[mask_bool, 1] * (1 - alpha)
        rgb[mask_bool, 2] = rgb[mask_bool, 2] * (1 - alpha)
    elif color == "green":
        rgb[mask_bool, 1] = 255
        rgb[mask_bool, 0] = rgb[mask_bool, 0] * (1 - alpha)
        rgb[mask_bool, 2] = rgb[mask_bool, 2] * (1 - alpha)
    elif color == "yellow":
        rgb[mask_bool, 0] = 255
        rgb[mask_bool, 1] = 255
        rgb[mask_bool, 2] = rgb[mask_bool, 2] * (1 - alpha)

    return rgb.clip(0, 255).astype(np.uint8)


def overlay_lung_and_nodule_on_ct_hu(
    ct_hu: np.ndarray,
    lung_mask2d: np.ndarray = None,
    nodule_mask2d: np.ndarray = None,
    lung_alpha=DEFAULT_LUNG_ALPHA,
    nodule_alpha=DEFAULT_NODULE_ALPHA,
    wl: float = DEFAULT_WINDOW_LEVEL,
    ww: float = DEFAULT_WINDOW_WIDTH
) -> np.ndarray:
    base = hu_window_to_uint8(ct_hu, wl=wl, ww=ww)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    if lung_mask2d is not None:
        lung_bool = lung_mask2d > 0
        rgb[lung_bool, 1] = 255
        rgb[lung_bool, 0] = rgb[lung_bool, 0] * (1 - lung_alpha)
        rgb[lung_bool, 2] = rgb[lung_bool, 2] * (1 - lung_alpha)

    if nodule_mask2d is not None:
        nodule_bool = nodule_mask2d > 0
        rgb[nodule_bool, 0] = 255
        rgb[nodule_bool, 1] = rgb[nodule_bool, 1] * (1 - nodule_alpha)
        rgb[nodule_bool, 2] = rgb[nodule_bool, 2] * (1 - nodule_alpha)

    return rgb.clip(0, 255).astype(np.uint8)


def apply_lung_mask_to_patch_hu(patch_hu: np.ndarray, lung_patch_mask: np.ndarray = None, outside_value: float = -1000.0):
    if lung_patch_mask is None:
        return patch_hu
    out = patch_hu.copy()
    out[lung_patch_mask <= 0] = outside_value
    return out


def draw_crosshair_on_rgb(rgb: np.ndarray, x: int, y: int, color=(255, 0, 0)) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape

    if 0 <= y < h:
        out[y, :, 0] = color[0]
        out[y, :, 1] = color[1]
        out[y, :, 2] = color[2]

    if 0 <= x < w:
        out[:, x, 0] = color[0]
        out[:, x, 1] = color[1]
        out[:, x, 2] = color[2]

    return out


def draw_center_marker_on_rgb(rgb: np.ndarray, x: int, y: int, color=(255, 255, 0), radius: int = 6) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape
    if not (0 <= x < w and 0 <= y < h):
        return out

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    ring = (dist2 <= radius ** 2) & (dist2 >= max(1, radius - 2) ** 2)
    out[ring, 0] = color[0]
    out[ring, 1] = color[1]
    out[ring, 2] = color[2]

    dot = dist2 <= 1
    out[dot, 0] = color[0]
    out[dot, 1] = color[1]
    out[dot, 2] = color[2]
    return out


def draw_polyline_on_rgb(rgb: np.ndarray, points, color=(0, 255, 255)):
    out = rgb.copy()
    if len(points) == 0:
        return out

    h, w, _ = out.shape

    for (x, y) in points:
        if 0 <= x < w and 0 <= y < h:
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            x1 = max(0, x - 1)
            x2 = min(w, x + 2)
            out[y1:y2, x1:x2, 0] = color[0]
            out[y1:y2, x1:x2, 1] = color[1]
            out[y1:y2, x1:x2, 2] = color[2]

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        out[ys[valid], xs[valid], 0] = color[0]
        out[ys[valid], xs[valid], 1] = color[1]
        out[ys[valid], xs[valid], 2] = color[2]

    return out

def polygon_to_mask(height: int, width: int, points):
    if len(points) < 3:
        return np.zeros((height, width), dtype=np.uint8)

    img = QImage(width, height, QImage.Format_Grayscale8)
    img.fill(0)

    polygon = QPolygon([QPoint(int(x), int(y)) for x, y in points])

    painter = QPainter(img)
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.setPen(QPen(QColor(255, 255, 255)))
    painter.drawPolygon(polygon)
    painter.end()

    ptr = img.bits()
    ptr.setsize(height * width)
    arr = np.frombuffer(ptr.asstring(height * width), dtype=np.uint8).reshape((height, width))
    return (arr > 0).astype(np.uint8)

def read_volume_from_dicom_dir(dicom_dir: Path):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"这个目录下没有检测到 DICOM 序列：{dicom_dir}")

    series_id = series_ids[0]
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()

    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    origin_xyz = np.array(img.GetOrigin(), dtype=np.float32)
    spacing_xyz = np.array(img.GetSpacing(), dtype=np.float32)
    direction = img.GetDirection()

    # 优先从 DICOM tag 里读取真正的 SeriesInstanceUID
    series_uid = None
    try:
        if reader.HasMetaDataKey(0, "0020|000e"):
            series_uid = reader.GetMetaData(0, "0020|000e").strip()
    except Exception:
        series_uid = None

    # 如果没读到，再退回 series_id / 文件夹名
    if not series_uid:
        series_uid = series_id if series_id else dicom_dir.name

    return {
        "image": img,
        "volume": vol,
        "origin_xyz": origin_xyz,
        "spacing_xyz": spacing_xyz,
        "direction": direction,
        "series_uid": series_uid,
        "source_type": "dicom",
        "display_name": str(dicom_dir),
    }


def read_volume_from_mhd_file(mhd_path: Path):
    img = sitk.ReadImage(str(mhd_path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)

    return {
        "image": img,
        "volume": vol,
        "origin_xyz": np.array(img.GetOrigin(), dtype=np.float32),
        "spacing_xyz": np.array(img.GetSpacing(), dtype=np.float32),
        "direction": img.GetDirection(),
        "series_uid": mhd_path.stem,
        "source_type": "mhd",
        "display_name": str(mhd_path),
    }

def find_dicom_series_dirs(root_dir: Path):
    series_dirs = []
    for p in root_dir.rglob("*"):
        if not p.is_dir():
            continue
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(p))
            if series_ids:
                series_dirs.append(p)
        except Exception:
            pass
    return sorted(series_dirs)

class ZoomableImageCanvas(QWidget):
    clicked = pyqtSignal(int, int)
    wheel_signal = pyqtSignal(int, bool, object)
    press_signal = pyqtSignal(int, int, int)
    move_signal = pyqtSignal(int, int, int)
    release_signal = pyqtSignal()

    def __init__(self, parent=None, allow_click=False, editable=False):
        super().__init__(parent)
        self.allow_click = allow_click
        self.editable = editable

        self.image_rgb = None
        self.qimage = None
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pressed_button = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 180)

    def set_image(self, rgb: np.ndarray):
        self.image_rgb = np.ascontiguousarray(rgb)
        h, w, _ = self.image_rgb.shape
        self.qimage = QImage(self.image_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        if self.zoom_factor <= 1.0:
            self.reset_view()
        self.update()

    def clear(self):
        self.image_rgb = None
        self.qimage = None
        self.reset_view()
        self.update()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def get_base_scale(self):
        if self.qimage is None or self.width() <= 1 or self.height() <= 1:
            return 1.0
        iw = self.qimage.width()
        ih = self.qimage.height()
        return min(self.width() / max(iw, 1), self.height() / max(ih, 1))

    def get_current_scale(self):
        return self.get_base_scale() * self.zoom_factor

    def get_draw_rect(self):
        if self.qimage is None:
            return QRectF()

        iw = self.qimage.width()
        ih = self.qimage.height()
        scale = self.get_current_scale()

        sw = iw * scale
        sh = ih * scale

        x = (self.width() - sw) / 2.0 + self.pan_x
        y = (self.height() - sh) / 2.0 + self.pan_y
        return QRectF(x, y, sw, sh)

    def clamp_pan(self):
        if self.qimage is None:
            self.pan_x = 0.0
            self.pan_y = 0.0
            return

        rect = self.get_draw_rect()
        sw = rect.width()
        sh = rect.height()

        base_x = (self.width() - sw) / 2.0
        base_y = (self.height() - sh) / 2.0

        if sw <= self.width():
            self.pan_x = 0.0
        else:
            min_pan_x = self.width() - sw - base_x
            max_pan_x = -base_x
            self.pan_x = max(min_pan_x, min(max_pan_x, self.pan_x))

        if sh <= self.height():
            self.pan_y = 0.0
        else:
            min_pan_y = self.height() - sh - base_y
            max_pan_y = -base_y
            self.pan_y = max(min_pan_y, min(max_pan_y, self.pan_y))

    def image_pos_from_widget_pos(self, px, py):
        if self.qimage is None:
            return None

        rect = self.get_draw_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None

        if not rect.contains(px, py):
            return None

        rx = (px - rect.x()) / rect.width()
        ry = (py - rect.y()) / rect.height()

        ix = int(rx * self.qimage.width())
        iy = int(ry * self.qimage.height())

        ix = max(0, min(self.qimage.width() - 1, ix))
        iy = max(0, min(self.qimage.height() - 1, iy))
        return ix, iy

    def apply_group_zoom(self, target_zoom, anchor_ratio_x=0.5, anchor_ratio_y=0.5):
        if self.qimage is None:
            return

        target_zoom = max(1.0, min(6.0, float(target_zoom)))
        old_rect = self.get_draw_rect()

        if old_rect.width() <= 0 or old_rect.height() <= 0:
            self.zoom_factor = target_zoom
            self.reset_view()
            self.update()
            return

        mx = old_rect.x() + anchor_ratio_x * old_rect.width()
        my = old_rect.y() + anchor_ratio_y * old_rect.height()

        old_zoom = self.zoom_factor
        if abs(target_zoom - old_zoom) < 1e-8:
            return

        ix = (mx - old_rect.x()) / old_rect.width()
        iy = (my - old_rect.y()) / old_rect.height()
        ix = float(np.clip(ix, 0.0, 1.0))
        iy = float(np.clip(iy, 0.0, 1.0))

        self.zoom_factor = target_zoom

        iw = self.qimage.width()
        ih = self.qimage.height()
        scale = self.get_current_scale()
        sw = iw * scale
        sh = ih * scale

        new_x = mx - ix * sw
        new_y = my - iy * sh

        center_x = (self.width() - sw) / 2.0
        center_y = (self.height() - sh) / 2.0

        self.pan_x = new_x - center_x
        self.pan_y = new_y - center_y

        self.clamp_pan()

        if self.zoom_factor <= 1.0:
            self.reset_view()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))

        if self.qimage is None:
            painter.setPen(QColor("#aaaaaa"))
            font = painter.font()
            font.setPointSize(max(10, font.pointSize()))
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载")
            return

        self.clamp_pan()
        rect = self.get_draw_rect()
        painter.drawImage(rect, self.qimage)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.clamp_pan()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1 if delta > 0 else -1
        ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
        self.wheel_signal.emit(step, ctrl_pressed, event.pos())

    def mousePressEvent(self, event):
        mapped = self.image_pos_from_widget_pos(event.pos().x(), event.pos().y())
        if mapped is None:
            return

        x_img, y_img = mapped

        if self.editable:
            if event.button() == Qt.LeftButton:
                self._pressed_button = Qt.LeftButton
                self.press_signal.emit(x_img, y_img, 1)
                return
            elif event.button() == Qt.RightButton:
                self._pressed_button = Qt.RightButton
                self.press_signal.emit(x_img, y_img, 0)
                return

        if self.allow_click and event.button() == Qt.LeftButton:
            self.clicked.emit(x_img, y_img)

    def mouseMoveEvent(self, event):
        if not self.editable:
            return
        if self._pressed_button is None:
            return

        mapped = self.image_pos_from_widget_pos(event.pos().x(), event.pos().y())
        if mapped is None:
            return

        x_img, y_img = mapped
        if self._pressed_button == Qt.LeftButton:
            self.move_signal.emit(x_img, y_img, 1)
        elif self._pressed_button == Qt.RightButton:
            self.move_signal.emit(x_img, y_img, 0)

    def mouseReleaseEvent(self, event):
        if self.editable:
            self._pressed_button = None
            self.release_signal.emit()


class ImageBlock(QWidget):
    def __init__(self, title_text: str, min_w=260, min_h=220, allow_click=False, editable=False, title_size=15):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.title = QLabel(title_text)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(
            f"font-size: {title_size}px; font-weight: bold; color: #f0f0f0; background: transparent;"
        )

        self.canvas = ZoomableImageCanvas(allow_click=allow_click, editable=editable)
        self.canvas.setMinimumSize(min_w, min_h)
        self.canvas.setStyleSheet("""
            QWidget {
                border: 2px solid #5c5f66;
                background: #111111;
                border-radius: 8px;
            }
        """)

        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

    def show_rgb(self, img_rgb: np.ndarray):
        self.canvas.set_image(img_rgb)

    def clear(self):
        self.canvas.clear()


class CTDirectInferFolderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ===== 自动选择最新训练 run =====
        exp_dir = config.RUNS_DIR / config.EXP_NAME
        has_valid_run = False

        if exp_dir.exists():
            run_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime)
            if run_dirs:
                latest_run = run_dirs[-1]
                config.set_run_dir(latest_run)
                has_valid_run = True
                logger.info("GUI 自动使用最新 run: %s", latest_run)
            else:
                logger.warning("未找到任何 run 目录，请确保先运行训练。")
        else:
            logger.warning("实验目录不存在: %s", exp_dir)

        self.screen = QApplication.primaryScreen().availableGeometry()
        self.screen_w = self.screen.width()

        if self.screen_w <= 1366:
            self.base_font = 12
            self.title_font = 13
            self.panel_width = 270
        elif self.screen_w <= 1600:
            self.base_font = 13
            self.title_font = 14
            self.panel_width = 285
        else:
            self.base_font = 15
            self.title_font = 16
            self.panel_width = 300

        self.setWindowTitle("肺小结节分割系统")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        best_weight_path = None
        if has_valid_run:
            best_weight_path = config.get_best_weight_path()

        if best_weight_path is None or (not best_weight_path.exists()):
            QMessageBox.warning(self, "警告", "未能自动找到最佳权重，请手动选择权重文件。")
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型权重文件", str(config.RUNS_DIR), "PyTorch Model (*.pth);;All Files (*)"
            )
            if not file_path:
                raise RuntimeError("未选择模型权重文件，无法启动 GUI。")
            best_weight_path = Path(file_path)

        self.model = UNet3D_Real(base_ch=16, use_cbam=config.USE_CBAM).to(self.device)
        try:
            ckpt = torch.load(best_weight_path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            self.model.load_state_dict(state_dict)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型权重失败：\n{e}")
            raise
        self.model.eval()

        self.ann_df = None
        self.load_annotations()

        self.ct_folder = None
        self.ct_file_list = []
        self.current_ct_index = -1

        self.ct_file = None
        self.series_uid = None
        self.source_type = None
        self.ct_volume = None
        self.ct_image = None
        self.ct_origin_xyz = None
        self.ct_spacing_xyz = None
        self.ct_direction = None
        self.lung_mask = None

        self.case_df = None
        self.current_nodule_index = -1

        self.current_center_xyz = None
        self.current_center_zyx = None
        self.current_diameter_mm = None

        self.manual_mode = False

        self.patch = None
        self.gt_mask = None
        self.pred_prob = None
        self.pred_mask_raw = None
        self.edited_mask = None
        self.current_dice = None
        self.raw_dice = None

        self.whole_ct_mask = None

        self.axial_z = 0
        self.coronal_y = 0
        self.sagittal_x = 0
        self.patch_z = config.PATCH_SIZE // 2

        self.undo_stack = []
        self.max_history = 30

        self.poly_points = []
        self.is_brush_drawing = False

        self.ct_blocks = []
        self.patch_blocks = []

        self.window_level = DEFAULT_WINDOW_LEVEL
        self.window_width = DEFAULT_WINDOW_WIDTH

        self.lung_alpha = DEFAULT_LUNG_ALPHA
        self.nodule_alpha = DEFAULT_NODULE_ALPHA
        self.patch_gt_alpha = DEFAULT_PATCH_GT_ALPHA
        self.patch_pred_alpha = DEFAULT_PATCH_PRED_ALPHA

        self.init_ui()

    def build_info_card(self):
        card = QFrame()
        card.setObjectName("InfoCard")
        outer_layout = QVBoxLayout(card)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        title = QLabel("当前信息")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: {self.base_font + 1}px; font-weight: bold; color:#f0f0f0;")
        outer_layout.addWidget(title)

        self.info_scroll = QScrollArea()
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.info_scroll.setMinimumHeight(120)
        self.info_scroll.setMaximumHeight(145)
        self.info_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #20242a;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #5f6874;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #707a87;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        info_inner = QWidget()
        info_layout = QVBoxLayout(info_inner)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(5)

        self.info_label = QLabel("请先打开CT。")
        self.info_label.setWordWrap(True)

        self.file_label = QLabel("当前文件：未加载")
        self.file_label.setWordWrap(True)

        self.metric_label = QLabel("当前 Dice: -")
        self.metric_label.setStyleSheet(f"font-size: {self.base_font + 3}px; font-weight: bold; color:#ff6666;")

        self.raw_metric_label = QLabel("原始 Dice: -")
        self.raw_metric_label.setStyleSheet(f"font-size: {self.base_font + 1}px; font-weight: bold; color:#66aaff;")

        self.center_label = QLabel("结节中心: -")
        self.center_label.setWordWrap(True)

        self.slice_label = QLabel("patch_z: - | axial_z: - | coronal_y: - | sagittal_x: -")
        self.slice_label.setWordWrap(True)

        self.progress_label = QLabel("CT进度: -")
        self.mode_label = QLabel("当前中心来源: -")
        self.mode_label.setStyleSheet(f"font-size: {self.base_font + 1}px; font-weight: bold; color:#55dd55;")

        self.whole_mask_label = QLabel("整CT mask 体素数: -")
        self.whole_mask_label.setStyleSheet(f"font-size: {self.base_font + 1}px; font-weight: bold; color:#ffbb55;")

        self.poly_label = QLabel("当前多边形点数: 0")
        self.poly_label.setStyleSheet(f"font-size: {self.base_font + 1}px; font-weight: bold; color:#55ccff;")

        for w in [
            self.info_label, self.file_label, self.progress_label, self.mode_label,
            self.metric_label, self.raw_metric_label, self.whole_mask_label,
            self.poly_label, self.center_label, self.slice_label
        ]:
            info_layout.addWidget(w)

        self.info_scroll.setWidget(info_inner)
        outer_layout.addWidget(self.info_scroll)

        return card

    def init_ui(self):
        root = QWidget()
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(14)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        self.axial_block = ImageBlock("Axial", allow_click=True, title_size=self.title_font)
        self.coronal_block = ImageBlock("Coronal", allow_click=True, title_size=self.title_font)
        self.sagittal_block = ImageBlock("Sagittal", allow_click=True, title_size=self.title_font)
        self.ct_blocks = [self.axial_block, self.coronal_block, self.sagittal_block]
        row1.addWidget(self.axial_block, 1)
        row1.addWidget(self.coronal_block, 1)
        row1.addWidget(self.sagittal_block, 1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.patch_block = ImageBlock("Patch", title_size=self.title_font)
        self.gt_block = ImageBlock("Patch + GT", title_size=self.title_font)
        self.pred_block = ImageBlock("Patch + Pred", editable=True, title_size=self.title_font)
        self.patch_blocks = [self.patch_block, self.gt_block, self.pred_block]
        row2.addWidget(self.patch_block, 1)
        row2.addWidget(self.gt_block, 1)
        row2.addWidget(self.pred_block, 1)

        left_layout.addLayout(row1, 1)
        left_layout.addLayout(row2, 1)
        left_layout.addWidget(self.build_info_card())
        left_layout.setStretch(0, 4)
        left_layout.setStretch(1, 4)
        left_layout.setStretch(2, 1)

        self.axial_block.canvas.clicked.connect(self.on_click_axial_view)
        self.coronal_block.canvas.clicked.connect(self.on_click_coronal_view)
        self.sagittal_block.canvas.clicked.connect(self.on_click_sagittal_view)

        self.axial_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_ct_wheel("axial", step, ctrl, pos))
        self.coronal_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_ct_wheel("coronal", step, ctrl, pos))
        self.sagittal_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_ct_wheel("sagittal", step, ctrl, pos))

        self.patch_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_patch_wheel(self.patch_block, step, ctrl, pos))
        self.gt_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_patch_wheel(self.gt_block, step, ctrl, pos))
        self.pred_block.canvas.wheel_signal.connect(lambda step, ctrl, pos: self.handle_patch_wheel(self.pred_block, step, ctrl, pos))

        self.pred_block.canvas.press_signal.connect(self.on_editor_press)
        self.pred_block.canvas.move_signal.connect(self.on_editor_move)
        self.pred_block.canvas.release_signal.connect(self.on_editor_release)

        # 右侧可滚动控制面板
        right_container = QWidget()
        right_container.setFixedWidth(self.panel_width)

        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 2, 10, 8)

        title = QLabel("控制面板")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: {self.base_font + 5}px; font-weight: bold;")
        right_layout.addWidget(title)

        self.open_ct_button = QPushButton("打开CT数据")
        self.open_ct_button.clicked.connect(self.open_ct_data_dispatcher)
        right_layout.addWidget(self.open_ct_button)

        ct_nav_row = QHBoxLayout()
        ct_nav_row.setSpacing(8)
        self.prev_ct_button = QPushButton("上一个CT")
        self.prev_ct_button.clicked.connect(self.prev_ct)
        self.next_ct_button = QPushButton("下一个CT")
        self.next_ct_button.clicked.connect(self.next_ct)
        ct_nav_row.addWidget(self.prev_ct_button)
        ct_nav_row.addWidget(self.next_ct_button)
        right_layout.addLayout(ct_nav_row)

        nodule_nav_row = QHBoxLayout()
        nodule_nav_row.setSpacing(8)
        self.prev_nodule_button = QPushButton("上一个结节")
        self.prev_nodule_button.clicked.connect(self.prev_nodule)
        self.next_nodule_button = QPushButton("下一个结节")
        self.next_nodule_button.clicked.connect(self.next_nodule)
        nodule_nav_row.addWidget(self.prev_nodule_button)
        nodule_nav_row.addWidget(self.next_nodule_button)
        right_layout.addLayout(nodule_nav_row)

        form_card = QFrame()
        form_card.setObjectName("InfoCard")
        form_layout = QFormLayout(form_card)
        form_layout.setSpacing(8)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.nodule_combo = QComboBox()
        self.nodule_combo.currentIndexChanged.connect(self.on_nodule_changed)
        form_layout.addRow("当前结节", self.nodule_combo)

        patch_thr_widget = QWidget()
        patch_thr_layout = QHBoxLayout(patch_thr_widget)
        patch_thr_layout.setContentsMargins(0, 0, 0, 0)
        patch_thr_layout.setSpacing(6)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(32, 128)
        self.patch_size_spin.setSingleStep(16)
        self.patch_size_spin.setValue(config.PATCH_SIZE)
        self.patch_size_spin.valueChanged.connect(self.on_patch_param_changed)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(config.DEFAULT_THRESHOLD)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)

        patch_thr_layout.addWidget(QLabel("Patch"))
        patch_thr_layout.addWidget(self.patch_size_spin, 1)
        patch_thr_layout.addWidget(QLabel("阈值"))
        patch_thr_layout.addWidget(self.threshold_spin, 1)
        form_layout.addRow("Patch/阈值", patch_thr_widget)

        brush_dia_widget = QWidget()
        brush_dia_layout = QHBoxLayout(brush_dia_widget)
        brush_dia_layout.setContentsMargins(0, 0, 0, 0)
        brush_dia_layout.setSpacing(6)

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 20)
        self.brush_spin.setValue(2)
        self.brush_spin.valueChanged.connect(self.on_brush_changed)

        self.manual_diameter_spin = QDoubleSpinBox()
        self.manual_diameter_spin.setRange(1.0, 30.0)
        self.manual_diameter_spin.setSingleStep(0.5)
        self.manual_diameter_spin.setValue(6.0)
        self.manual_diameter_spin.valueChanged.connect(self.on_manual_diameter_changed)

        brush_dia_layout.addWidget(QLabel("画笔"))
        brush_dia_layout.addWidget(self.brush_spin, 1)
        brush_dia_layout.addWidget(QLabel("直径"))
        brush_dia_layout.addWidget(self.manual_diameter_spin, 1)
        form_layout.addRow("画笔/手动直径", brush_dia_widget)

        self.edit_mode_combo = QComboBox()
        self.edit_mode_combo.addItems([
            "刷子模式（左键补/ 右键擦）",
            "多边形补",
            "多边形擦",
        ])
        form_layout.addRow("编辑模式", self.edit_mode_combo)

        wl_ww_widget = QWidget()
        wl_ww_layout = QHBoxLayout(wl_ww_widget)
        wl_ww_layout.setContentsMargins(0, 0, 0, 0)
        wl_ww_layout.setSpacing(6)

        self.wl_spin = QSpinBox()
        self.wl_spin.setRange(-1500, 500)
        self.wl_spin.setSingleStep(10)
        self.wl_spin.setValue(int(self.window_level))
        self.wl_spin.valueChanged.connect(self.on_window_value_changed)

        self.ww_spin = QSpinBox()
        self.ww_spin.setRange(100, 3000)
        self.ww_spin.setSingleStep(50)
        self.ww_spin.setValue(int(self.window_width))
        self.ww_spin.valueChanged.connect(self.on_window_value_changed)

        wl_ww_layout.addWidget(QLabel("WL"))
        wl_ww_layout.addWidget(self.wl_spin, 1)
        wl_ww_layout.addWidget(QLabel("WW"))
        wl_ww_layout.addWidget(self.ww_spin, 1)
        form_layout.addRow("窗位/窗宽", wl_ww_widget)

        alpha_widget = QWidget()
        alpha_layout = QHBoxLayout(alpha_widget)
        alpha_layout.setContentsMargins(0, 0, 0, 0)
        alpha_layout.setSpacing(6)

        self.lung_alpha_spin = QDoubleSpinBox()
        self.lung_alpha_spin.setRange(0.0, 1.0)
        self.lung_alpha_spin.setSingleStep(0.05)
        self.lung_alpha_spin.setValue(self.lung_alpha)
        self.lung_alpha_spin.valueChanged.connect(self.on_alpha_value_changed)

        self.nodule_alpha_spin = QDoubleSpinBox()
        self.nodule_alpha_spin.setRange(0.0, 1.0)
        self.nodule_alpha_spin.setSingleStep(0.05)
        self.nodule_alpha_spin.setValue(self.nodule_alpha)
        self.nodule_alpha_spin.valueChanged.connect(self.on_alpha_value_changed)

        alpha_layout.addWidget(QLabel("肺"))
        alpha_layout.addWidget(self.lung_alpha_spin, 1)
        alpha_layout.addWidget(QLabel("结节"))
        alpha_layout.addWidget(self.nodule_alpha_spin, 1)
        form_layout.addRow("CT透明度", alpha_widget)

        patch_alpha_widget = QWidget()
        patch_alpha_layout = QHBoxLayout(patch_alpha_widget)
        patch_alpha_layout.setContentsMargins(0, 0, 0, 0)
        patch_alpha_layout.setSpacing(6)

        self.patch_gt_alpha_spin = QDoubleSpinBox()
        self.patch_gt_alpha_spin.setRange(0.0, 1.0)
        self.patch_gt_alpha_spin.setSingleStep(0.05)
        self.patch_gt_alpha_spin.setValue(self.patch_gt_alpha)
        self.patch_gt_alpha_spin.valueChanged.connect(self.on_alpha_value_changed)

        self.patch_pred_alpha_spin = QDoubleSpinBox()
        self.patch_pred_alpha_spin.setRange(0.0, 1.0)
        self.patch_pred_alpha_spin.setSingleStep(0.05)
        self.patch_pred_alpha_spin.setValue(self.patch_pred_alpha)
        self.patch_pred_alpha_spin.valueChanged.connect(self.on_alpha_value_changed)

        patch_alpha_layout.addWidget(QLabel("GT"))
        patch_alpha_layout.addWidget(self.patch_gt_alpha_spin, 1)
        patch_alpha_layout.addWidget(QLabel("Pred"))
        patch_alpha_layout.addWidget(self.patch_pred_alpha_spin, 1)
        form_layout.addRow("Patch透明度", patch_alpha_widget)

        right_layout.addWidget(form_card)

        restore_row = QHBoxLayout()
        restore_row.setSpacing(8)
        self.reset_window_button = QPushButton("还原窗位窗宽")
        self.reset_window_button.clicked.connect(self.reset_window_settings)

        self.reset_alpha_button = QPushButton("还原透明度")
        self.reset_alpha_button.clicked.connect(self.reset_alpha_settings)

        restore_row.addWidget(self.reset_window_button)
        restore_row.addWidget(self.reset_alpha_button)
        right_layout.addLayout(restore_row)

        self.back_to_nodule_button = QPushButton("回到当前标注中心")
        self.back_to_nodule_button.clicked.connect(self.back_to_current_nodule_center)
        right_layout.addWidget(self.back_to_nodule_button)

        self.run_button = QPushButton("运行分割")
        self.run_button.clicked.connect(self.run_crop_and_infer)
        right_layout.addWidget(self.run_button)

        poly_row = QHBoxLayout()
        poly_row.setSpacing(8)
        self.fill_polygon_button = QPushButton("执行多边形补/擦")
        self.fill_polygon_button.clicked.connect(self.apply_polygon_by_mode)
        self.clear_polygon_button = QPushButton("清空多边形点")
        self.clear_polygon_button.clicked.connect(self.clear_polygon_points)
        poly_row.addWidget(self.fill_polygon_button)
        poly_row.addWidget(self.clear_polygon_button)
        right_layout.addLayout(poly_row)

        self.undo_button = QPushButton("撤销")
        self.undo_button.clicked.connect(self.undo_edit)
        right_layout.addWidget(self.undo_button)

        self.reset_button = QPushButton("重置为模型原始预测")
        self.reset_button.clicked.connect(self.reset_edited_mask)
        right_layout.addWidget(self.reset_button)

        self.apply_patch_button = QPushButton("把当前修正结果回贴到整CT")
        self.apply_patch_button.clicked.connect(self.apply_current_patch_to_whole_ct)
        right_layout.addWidget(self.apply_patch_button)

        self.clear_whole_mask_button = QPushButton("清空整CT累计 mask")
        self.clear_whole_mask_button.clicked.connect(self.clear_whole_ct_mask)
        right_layout.addWidget(self.clear_whole_mask_button)

        self.save_patch_mask_button = QPushButton("保存修正后 Patch Mask")
        self.save_patch_mask_button.clicked.connect(self.save_current_mask_npy)
        right_layout.addWidget(self.save_patch_mask_button)

        self.save_whole_ct_mask_button = QPushButton("保存整CT Mask")
        self.save_whole_ct_mask_button.clicked.connect(self.save_current_whole_ct_mask_mhd)
        right_layout.addWidget(self.save_whole_ct_mask_button)

        self.save_button = QPushButton("保存当前预测叠加图")
        self.save_button.clicked.connect(self.save_current_pred_png)
        right_layout.addWidget(self.save_button)

        right_layout.addStretch()

        # 外层滚动区
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_container)
        right_scroll.setFixedWidth(self.panel_width + 20)
        right_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #20242a;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #5f6874;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #707a87;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(right_scroll, 0)

        self.setCentralWidget(root)
        self.statusBar().showMessage("就绪")

    def load_annotations(self):
        ann_path = Path(config.LUNA_ANN_CSV)
        if not ann_path.exists():
            QMessageBox.warning(self, "提示", f"annotations.csv 不存在：\n{ann_path}")
            self.ann_df = None
            return
        try:
            self.ann_df = pd.read_csv(ann_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取 annotations.csv 失败：\n{e}")
            self.ann_df = None
    
    def world_xyz_to_voxel_zyx(self, x_world: float, y_world: float, z_world: float):
        if self.ct_image is None:
            raise RuntimeError("CT 尚未加载")

        idx_xyz = self.ct_image.TransformPhysicalPointToIndex(
            (float(x_world), float(y_world), float(z_world))
        )
        idx_xyz = np.array(idx_xyz, dtype=int)
        idx_zyx = idx_xyz[::-1]
        return idx_xyz, idx_zyx

    def open_ct_data_dispatcher(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("选择打开方式")
        msg.setText("请选择CT数据打开方式")
        msg.setIcon(QMessageBox.Question)

        btn_mhd_folder = msg.addButton("选择MHD文件夹", QMessageBox.ActionRole)
        btn_single_mhd = msg.addButton("打开单个MHD", QMessageBox.ActionRole)
        btn_dicom_series = msg.addButton("打开一个DICOM序列文件夹", QMessageBox.ActionRole)
        btn_dicom_root = msg.addButton("打开DICOM根目录", QMessageBox.ActionRole)
        msg.addButton("取消", QMessageBox.RejectRole)

        msg.exec_()
        clicked = msg.clickedButton()

        if clicked == btn_mhd_folder:
            self.open_ct_folder()
        elif clicked == btn_single_mhd:
            self.open_single_ct()
        elif clicked == btn_dicom_series:
            self.open_single_dicom_series_dir()
        elif clicked == btn_dicom_root:
            self.open_dicom_root_folder()

    def on_window_value_changed(self):
        self.window_level = float(self.wl_spin.value())
        self.window_width = float(self.ww_spin.value())
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def reset_window_settings(self):
        self.wl_spin.blockSignals(True)
        self.ww_spin.blockSignals(True)
        self.wl_spin.setValue(int(DEFAULT_WINDOW_LEVEL))
        self.ww_spin.setValue(int(DEFAULT_WINDOW_WIDTH))
        self.wl_spin.blockSignals(False)
        self.ww_spin.blockSignals(False)
        self.window_level = DEFAULT_WINDOW_LEVEL
        self.window_width = DEFAULT_WINDOW_WIDTH
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()
        self.statusBar().showMessage("已还原默认窗位窗宽", 2000)

    def on_alpha_value_changed(self):
        self.lung_alpha = float(self.lung_alpha_spin.value())
        self.nodule_alpha = float(self.nodule_alpha_spin.value())
        self.patch_gt_alpha = float(self.patch_gt_alpha_spin.value())
        self.patch_pred_alpha = float(self.patch_pred_alpha_spin.value())
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def reset_alpha_settings(self):
        widgets = [
            (self.lung_alpha_spin, DEFAULT_LUNG_ALPHA),
            (self.nodule_alpha_spin, DEFAULT_NODULE_ALPHA),
            (self.patch_gt_alpha_spin, DEFAULT_PATCH_GT_ALPHA),
            (self.patch_pred_alpha_spin, DEFAULT_PATCH_PRED_ALPHA),
        ]
        for w, v in widgets:
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

        self.lung_alpha = DEFAULT_LUNG_ALPHA
        self.nodule_alpha = DEFAULT_NODULE_ALPHA
        self.patch_gt_alpha = DEFAULT_PATCH_GT_ALPHA
        self.patch_pred_alpha = DEFAULT_PATCH_PRED_ALPHA

        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()
        self.statusBar().showMessage("已还原默认透明度", 2000)

    def on_patch_param_changed(self):
        v = int(self.patch_size_spin.value())
        if v % 2 != 0:
            self.patch_size_spin.blockSignals(True)
            self.patch_size_spin.setValue(v + 1)
            self.patch_size_spin.blockSignals(False)
        self.update_texts()

    def on_brush_changed(self):
        self.update_texts()

    def on_manual_diameter_changed(self):
        if self.manual_mode:
            self.current_diameter_mm = float(self.manual_diameter_spin.value())
            self.update_texts()

    def group_zoom(self, blocks, source_block, step, mouse_pos):
        source_canvas = source_block.canvas
        new_zoom = source_canvas.zoom_factor + 0.18 * step
        new_zoom = max(1.0, min(6.0, new_zoom))

        if source_canvas.qimage is not None:
            old_rect = source_canvas.get_draw_rect()
            if old_rect.width() > 0 and old_rect.height() > 0:
                anchor_x = float(np.clip((mouse_pos.x() - old_rect.x()) / old_rect.width(), 0.0, 1.0))
                anchor_y = float(np.clip((mouse_pos.y() - old_rect.y()) / old_rect.height(), 0.0, 1.0))
            else:
                anchor_x, anchor_y = 0.5, 0.5
        else:
            anchor_x, anchor_y = 0.5, 0.5

        for block in blocks:
            block.canvas.apply_group_zoom(new_zoom, anchor_x, anchor_y)

    def get_edit_mode(self):
        return self.edit_mode_combo.currentText()

    def is_brush_mode(self):
        return self.get_edit_mode() == "刷子模式（左键补/ 右键擦）"

    def is_polygon_mode(self):
        return self.get_edit_mode() in ("多边形补", "多边形擦")

    def handle_ct_wheel(self, view_name: str, step: int, ctrl_pressed: bool, mouse_pos):
        if ctrl_pressed:
            block = {
                "axial": self.axial_block,
                "coronal": self.coronal_block,
                "sagittal": self.sagittal_block,
            }[view_name]
            self.group_zoom(self.ct_blocks, block, step, mouse_pos)
            return

        if self.ct_volume is None:
            return

        Z, Y, X = self.ct_volume.shape
        if view_name == "axial":
            self.axial_z = int(np.clip(self.axial_z + step, 0, Z - 1))
        elif view_name == "coronal":
            self.coronal_y = int(np.clip(self.coronal_y + step, 0, Y - 1))
        else:
            self.sagittal_x = int(np.clip(self.sagittal_x + step, 0, X - 1))

        self.refresh_ct_views()
        self.update_texts()

    def handle_patch_wheel(self, block: ImageBlock, step: int, ctrl_pressed: bool, mouse_pos):
        if ctrl_pressed:
            self.group_zoom(self.patch_blocks, block, step, mouse_pos)
            return

        if self.patch is None:
            return

        self.patch_z = int(np.clip(self.patch_z + step, 0, self.patch.shape[0] - 1))
        self.poly_points = []
        self.refresh_patch_views()
        self.update_texts()

    def open_ct_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择MHD文件夹", str(config.RAW_DIR))
        if not folder:
            return

        self.ct_folder = Path(folder)
        self.ct_file_list = sorted([str(p) for p in self.ct_folder.rglob("*.mhd")])
        self.current_ct_index = -1

        if len(self.ct_file_list) == 0:
            QMessageBox.warning(self, "提示", "这个文件夹里没有找到 .mhd 文件。")
            return

        self.current_ct_index = 0
        self.load_ct_by_index(self.current_ct_index)

    def open_single_ct(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择MHD文件",
            str(config.RAW_DIR),
            "MHD Files (*.mhd);;All Files (*)"
        )
        if not file_path:
            return

        p = Path(file_path)
        parent = p.parent
        self.ct_folder = parent
        self.ct_file_list = sorted([str(x) for x in parent.glob("*.mhd")])

        if str(p) in self.ct_file_list:
            self.current_ct_index = self.ct_file_list.index(str(p))
        else:
            self.ct_file_list = [str(p)]
            self.current_ct_index = 0

        self.load_ct_by_index(self.current_ct_index)

    def open_single_dicom_series_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择一个 DICOM 序列文件夹", str(config.RAW_DIR))
        if not folder:
            return

        dicom_dir = Path(folder)
        self.ct_folder = dicom_dir.parent
        self.ct_file_list = [str(dicom_dir)]
        self.current_ct_index = 0
        self.load_ct_by_index(self.current_ct_index)

    def open_dicom_root_folder(self):
        root = QFileDialog.getExistingDirectory(self, "选择 DICOM 根目录", str(config.RAW_DIR))
        if not root:
            return

        root_dir = Path(root)
        series_dirs = find_dicom_series_dirs(root_dir)

        if len(series_dirs) == 0:
            QMessageBox.warning(self, "提示", "没有在该目录下找到 DICOM 序列。")
            return

        self.ct_folder = root_dir
        self.ct_file_list = [str(d) for d in series_dirs]
        self.current_ct_index = 0
        self.load_ct_by_index(self.current_ct_index)

    def load_ct_by_index(self, index: int):
        if index < 0 or index >= len(self.ct_file_list):
            QMessageBox.warning(self, "提示", f"CT索引越界: {index}")
            return
        self.load_ct_file(self.ct_file_list[index])

    def clear_edit_history(self):
        self.undo_stack.clear()

    def clear_current_patch_results(self):
        self.patch = None
        self.gt_mask = None
        self.pred_prob = None
        self.pred_mask_raw = None
        self.edited_mask = None
        self.current_dice = None
        self.raw_dice = None
        self.patch_z = config.PATCH_SIZE // 2
        self.metric_label.setText("当前 Dice: -")
        self.raw_metric_label.setText("原始 Dice: -")
        self.clear_edit_history()
        self.poly_points = []
        self.is_brush_drawing = False

    def reset_group_zoom(self):
        for b in self.ct_blocks + self.patch_blocks:
            b.canvas.reset_view()
            b.canvas.update()

    def load_ct_file(self, file_path: str):
        try:
            p = Path(file_path)

            if p.is_dir():
                info = read_volume_from_dicom_dir(p)
            elif p.suffix.lower() == ".mhd":
                info = read_volume_from_mhd_file(p)
            else:
                raise RuntimeError(f"不支持的输入：{p}")

            self.ct_file = info["display_name"]
            self.ct_image = info["image"]
            self.series_uid = info["series_uid"]
            self.source_type = info["source_type"]
            self.ct_volume = info["volume"]
            self.ct_origin_xyz = info["origin_xyz"]
            self.ct_spacing_xyz = info["spacing_xyz"]
            self.ct_direction = info["direction"]

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.statusBar().showMessage("正在进行肺实质分割，请稍候...")
                QApplication.processEvents()

                self.lung_mask = segment_lung_mask(self.ct_volume)

            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar().showMessage("肺实质分割完成", 3000)

            self.whole_ct_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)

            Z, Y, X = self.ct_volume.shape
            self.axial_z = Z // 2
            self.coronal_y = Y // 2
            self.sagittal_x = X // 2

            self.clear_current_patch_results()
            self.manual_mode = False
            self.reset_group_zoom()

            self.reload_case_nodules()
            self.refresh_ct_views()
            self.refresh_patch_views()
            self.update_texts()

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def prev_ct(self):
        if not self.ct_file_list:
            QMessageBox.information(self, "提示", "请先选择CT。")
            return
        if self.current_ct_index <= 0:
            QMessageBox.information(self, "提示", "已经是第一个CT了。")
            return
        self.current_ct_index -= 1
        self.load_ct_by_index(self.current_ct_index)

    def next_ct(self):
        if not self.ct_file_list:
            QMessageBox.information(self, "提示", "请先选择CT。")
            return
        if self.current_ct_index >= len(self.ct_file_list) - 1:
            QMessageBox.information(self, "提示", "已经是最后一个CT了。")
            return
        self.current_ct_index += 1
        self.load_ct_by_index(self.current_ct_index)

    def reload_case_nodules(self):
        if self.ct_file is None or self.ann_df is None:
            return

        self.case_df = self.ann_df[self.ann_df["seriesuid"] == self.series_uid].copy().reset_index(drop=True)

        self.nodule_combo.blockSignals(True)
        self.nodule_combo.clear()

        if len(self.case_df) == 0:
            self.current_nodule_index = -1
            self.current_center_xyz = None
            self.current_center_zyx = None
            self.current_diameter_mm = None
            self.manual_mode = True
            self.nodule_combo.addItem("无标注结节：请点击三视图手动指定中心")
            self.nodule_combo.blockSignals(False)

            self.clear_current_patch_results()
            self.refresh_patch_views()
            self.update_texts()
            return

        for i, row in self.case_df.iterrows():
            d = float(row["diameter_mm"])
            x, y, z = row["coordX"], row["coordY"], row["coordZ"]
            self.nodule_combo.addItem(f"nodule#{i} | d={d:.2f}mm | world=({x:.1f},{y:.1f},{z:.1f})")

        self.manual_mode = False
        self.nodule_combo.blockSignals(False)
        self.nodule_combo.setCurrentIndex(0)
        self.on_nodule_changed()

    def prev_nodule(self):
        if self.case_df is None or len(self.case_df) == 0:
            QMessageBox.information(self, "提示", "当前CT没有标注结节。")
            return
        idx = self.nodule_combo.currentIndex()
        if idx <= 0:
            QMessageBox.information(self, "提示", "已经是第一个结节了。")
            return
        self.nodule_combo.setCurrentIndex(idx - 1)

    def next_nodule(self):
        if self.case_df is None or len(self.case_df) == 0:
            QMessageBox.information(self, "提示", "当前CT没有标注结节。")
            return
        idx = self.nodule_combo.currentIndex()
        if idx >= len(self.case_df) - 1:
            QMessageBox.information(self, "提示", "已经是最后一个结节了。")
            return
        self.nodule_combo.setCurrentIndex(idx + 1)

    def on_nodule_changed(self):
        if self.case_df is None or len(self.case_df) == 0:
            return

        idx = self.nodule_combo.currentIndex()
        if idx < 0 or idx >= len(self.case_df):
            return

        self.current_nodule_index = idx
        row = self.case_df.iloc[idx]

        x_world = float(row["coordX"])
        y_world = float(row["coordY"])
        z_world = float(row["coordZ"])
        self.current_diameter_mm = float(row["diameter_mm"])

        voxel_xyz, voxel_zyx = self.world_xyz_to_voxel_zyx(x_world, y_world, z_world)
        self.current_center_xyz = voxel_xyz
        self.current_center_zyx = voxel_zyx
        self.manual_mode = False

        self.axial_z = int(self.current_center_zyx[0])
        self.coronal_y = int(self.current_center_zyx[1])
        self.sagittal_x = int(self.current_center_zyx[2])

        self.clear_current_patch_results()
        self.refresh_patch_views()
        self.refresh_ct_views()
        self.update_texts()

    def back_to_current_nodule_center(self):
        if self.case_df is None or len(self.case_df) == 0:
            QMessageBox.information(self, "提示", "当前CT没有标注结节。")
            return

        idx = self.nodule_combo.currentIndex()
        if idx < 0 or idx >= len(self.case_df):
            QMessageBox.information(self, "提示", "当前没有有效的标注结节。")
            return

        row = self.case_df.iloc[idx]
        x_world = float(row["coordX"])
        y_world = float(row["coordY"])
        z_world = float(row["coordZ"])
        self.current_diameter_mm = float(row["diameter_mm"])

        voxel_xyz, voxel_zyx = self.world_xyz_to_voxel_zyx(x_world, y_world, z_world)
        self.current_center_xyz = voxel_xyz
        self.current_center_zyx = voxel_zyx
        self.manual_mode = False

        self.axial_z = int(self.current_center_zyx[0])
        self.coronal_y = int(self.current_center_zyx[1])
        self.sagittal_x = int(self.current_center_zyx[2])

        self.clear_current_patch_results()
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def use_crosshair_as_current_center(self):
        self.current_center_zyx = np.array([self.axial_z, self.coronal_y, self.sagittal_x], dtype=int)
        self.current_center_xyz = self.current_center_zyx[::-1].copy()
        self.current_diameter_mm = float(self.manual_diameter_spin.value())
        self.current_nodule_index = -1
        self.manual_mode = True

    def on_click_axial_view(self, x_img, y_img):
        if self.ct_volume is None:
            return
        self.coronal_y = int(y_img)
        self.sagittal_x = int(x_img)
        self.use_crosshair_as_current_center()
        self.clear_current_patch_results()
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def on_click_coronal_view(self, x_img, y_img):
        if self.ct_volume is None:
            return
        self.axial_z = int(y_img)
        self.sagittal_x = int(x_img)
        self.use_crosshair_as_current_center()
        self.clear_current_patch_results()
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def on_click_sagittal_view(self, x_img, y_img):
        if self.ct_volume is None:
            return
        self.axial_z = int(y_img)
        self.coronal_y = int(x_img)
        self.use_crosshair_as_current_center()
        self.clear_current_patch_results()
        self.refresh_ct_views()
        self.refresh_patch_views()
        self.update_texts()

    def on_threshold_changed(self):
        if self.patch is not None:
            self.clear_edit_history()
            self.poly_points = []
            self.run_inference_only()
        else:
            self.update_texts()

    def refresh_ct_views(self):
        if self.ct_volume is None:
            return

        lung_mask = self.lung_mask if self.lung_mask is not None else np.zeros_like(self.ct_volume, dtype=np.uint8)
        whole_mask = self.whole_ct_mask if self.whole_ct_mask is not None else np.zeros_like(self.ct_volume, dtype=np.uint8)

        axial = self.ct_volume[self.axial_z]
        axial_rgb = overlay_lung_and_nodule_on_ct_hu(
            axial,
            lung_mask2d=lung_mask[self.axial_z],
            nodule_mask2d=whole_mask[self.axial_z],
            lung_alpha=self.lung_alpha,
            nodule_alpha=self.nodule_alpha,
            wl=self.window_level,
            ww=self.window_width
        )
        axial_rgb = draw_crosshair_on_rgb(axial_rgb, self.sagittal_x, self.coronal_y, color=(255, 0, 0))
        axial_rgb = draw_center_marker_on_rgb(axial_rgb, self.sagittal_x, self.coronal_y, color=(255, 255, 0), radius=6)
        self.axial_block.show_rgb(axial_rgb)

        coronal = self.ct_volume[:, self.coronal_y, :]
        coronal_rgb = overlay_lung_and_nodule_on_ct_hu(
            coronal,
            lung_mask2d=lung_mask[:, self.coronal_y, :],
            nodule_mask2d=whole_mask[:, self.coronal_y, :],
            lung_alpha=self.lung_alpha,
            nodule_alpha=self.nodule_alpha,
            wl=self.window_level,
            ww=self.window_width
        )
        coronal_rgb = draw_crosshair_on_rgb(coronal_rgb, self.sagittal_x, self.axial_z, color=(0, 255, 0))
        coronal_rgb = draw_center_marker_on_rgb(coronal_rgb, self.sagittal_x, self.axial_z, color=(255, 255, 0), radius=6)
        self.coronal_block.show_rgb(coronal_rgb)

        sagittal = self.ct_volume[:, :, self.sagittal_x]
        sagittal_rgb = overlay_lung_and_nodule_on_ct_hu(
            sagittal,
            lung_mask2d=lung_mask[:, :, self.sagittal_x],
            nodule_mask2d=whole_mask[:, :, self.sagittal_x],
            lung_alpha=self.lung_alpha,
            nodule_alpha=self.nodule_alpha,
            wl=self.window_level,
            ww=self.window_width
        )
        sagittal_rgb = draw_crosshair_on_rgb(sagittal_rgb, self.coronal_y, self.axial_z, color=(0, 0, 255))
        sagittal_rgb = draw_center_marker_on_rgb(sagittal_rgb, self.coronal_y, self.axial_z, color=(255, 255, 0), radius=6)
        self.sagittal_block.show_rgb(sagittal_rgb)

    def run_crop_and_infer(self):
        if self.ct_volume is None:
            QMessageBox.warning(self, "提示", "请先打开CT。")
            return

        if self.current_center_zyx is None:
            QMessageBox.warning(self, "提示", "当前没有可用中心。请先选结节，或者点击CT图指定中心。")
            return

        patch_size = int(self.patch_size_spin.value())
        cz, cy, cx = self.current_center_zyx.tolist()

        if self.current_diameter_mm is None:
            self.current_diameter_mm = float(self.manual_diameter_spin.value())

        raw_patch = safe_crop_3d(self.ct_volume, cz, cy, cx, patch=patch_size).astype(np.float32)

        if self.lung_mask is not None:
            lung_patch = safe_crop_3d(self.lung_mask, cz, cy, cx, patch=patch_size).astype(np.uint8)
            self.patch = apply_lung_mask_to_patch_hu(raw_patch, lung_patch_mask=lung_patch, outside_value=-1000.0).astype(np.float32)
        else:
            self.patch = raw_patch

        spacing_zyx = np.array(self.ct_spacing_xyz[::-1], dtype=np.float32)

        self.gt_mask = make_sphere_mask(
            patch_size=patch_size,
            spacing_zyx=spacing_zyx,
            diameter_mm=self.current_diameter_mm
        ).astype(np.float32)

        self.patch_z = patch_size // 2
        self.clear_edit_history()
        self.poly_points = []
        for b in self.patch_blocks:
            b.canvas.reset_view()
            b.canvas.update()
        self.run_inference_only()

    def run_inference_only(self):
        if self.patch is None:
            return

        thr = float(self.threshold_spin.value())
        patch_n = minmax_norm(self.patch).astype(np.float32)
        self.pred_prob, pred_bin = predict_mask_from_patch(
            self.model,
            patch_n,
            self.device,
            threshold=thr,
        )

        pred_bin = pred_bin.astype(np.uint8)

        if config.POSTPROCESS_ENABLE:
            pred_bin = postprocess_nodule_mask(
                pred_bin,
                method=config.POSTPROCESS_METHOD,
                connectivity=config.POSTPROCESS_CONNECTIVITY,
                min_size=config.POSTPROCESS_MIN_SIZE,
            ).astype(np.uint8)

        self.pred_mask_raw = pred_bin.astype(np.float32)
        self.edited_mask = self.pred_mask_raw.copy()

        if self.gt_mask is not None:
            self.raw_dice = dice_score_np(self.pred_mask_raw, self.gt_mask)
            self.current_dice = dice_score_np(self.edited_mask, self.gt_mask)
            self.raw_metric_label.setText(f"原始 Dice: {self.raw_dice:.4f}")
            self.metric_label.setText(f"当前 Dice: {self.current_dice:.4f}")
        else:
            self.raw_metric_label.setText("原始 Dice: -")
            self.metric_label.setText("当前 Dice: -")

        self.refresh_patch_views()
        self.update_texts()

    def refresh_patch_views(self):
        if self.patch is None:
            self.patch_block.clear()
            self.gt_block.clear()
            self.pred_block.clear()
            return

        z = max(0, min(self.patch.shape[0] - 1, self.patch_z))
        patch_slice_hu = self.patch[z]

        patch_u8 = hu_window_to_uint8(
            patch_slice_hu,
            wl=self.window_level,
            ww=self.window_width
        )
        patch_rgb = np.stack([patch_u8] * 3, axis=-1)
        self.patch_block.show_rgb(patch_rgb)

        if self.gt_mask is not None:
            gt_overlay = overlay_mask_on_hu(
                patch_slice_hu,
                self.gt_mask[z],
                color="green",
                alpha=self.patch_gt_alpha,
                wl=self.window_level,
                ww=self.window_width
            )
            self.gt_block.show_rgb(gt_overlay)
        else:
            self.gt_block.clear()

        if self.edited_mask is not None:
            pred_overlay = overlay_mask_on_hu(
                patch_slice_hu,
                self.edited_mask[z],
                color="red",
                alpha=self.patch_pred_alpha,
                wl=self.window_level,
                ww=self.window_width
            )
            if len(self.poly_points) > 0:
                mode = self.get_edit_mode()
                poly_color = (0, 255, 255) if "补" in mode else (255, 255, 0)
                pred_overlay = draw_polyline_on_rgb(pred_overlay, self.poly_points, color=poly_color)
            self.pred_block.show_rgb(pred_overlay)
        else:
            self.pred_block.clear()

    def push_undo_state(self):
        if self.edited_mask is None:
            return
        self.undo_stack.append(self.edited_mask.copy())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

    def undo_edit(self):
        if not self.undo_stack:
            QMessageBox.information(self, "提示", "没有可以撤销的操作。")
            return
        self.edited_mask = self.undo_stack.pop()
        self.update_metric()
        self.refresh_patch_views()
        self.update_texts()

    def reset_edited_mask(self):
        if self.pred_mask_raw is None:
            QMessageBox.warning(self, "提示", "请先完成模型预测。")
            return

        self.push_undo_state()
        self.edited_mask = self.pred_mask_raw.copy()
        self.poly_points = []
        self.update_metric()
        self.refresh_patch_views()
        self.update_texts()
        QMessageBox.information(self, "提示", "已经恢复为模型原始预测。")

    def update_metric(self):
        if self.edited_mask is None or self.gt_mask is None:
            self.metric_label.setText("当前 Dice: -")
            return
        self.current_dice = dice_score_np(self.edited_mask, self.gt_mask)
        self.metric_label.setText(f"当前 Dice: {self.current_dice:.4f}")

    def make_circular_region(self, h, w, x, y, rr):
        yy, xx = np.ogrid[:h, :w]
        return ((xx - x) ** 2 + (yy - y) ** 2) <= rr ** 2

    def apply_brush(self, z, x, y, add_mode: bool):
        sl = self.edited_mask[z]
        h, w = sl.shape
        rr = int(self.brush_spin.value())
        region = self.make_circular_region(h, w, x, y, rr)

        if add_mode:
            sl[region] = 1.0
        else:
            sl[region] = 0.0

    def on_editor_press(self, x, y, button_mode):
        if self.edited_mask is None:
            return

        if self.is_brush_mode():
            self.push_undo_state()
            self.is_brush_drawing = True
            z = self.patch_z
            add_mode = (button_mode == 1)
            self.apply_brush(z, x, y, add_mode)
            self.update_metric()
            self.refresh_patch_views()
            self.update_texts()

        elif self.is_polygon_mode():
            if button_mode == 1:
                self.poly_points.append((int(x), int(y)))
            else:
                if len(self.poly_points) > 0:
                    self.poly_points.pop()
            self.refresh_patch_views()
            self.update_texts()

    def on_editor_move(self, x, y, button_mode):
        if self.edited_mask is None:
            return
        if not self.is_brush_mode():
            return
        if not self.is_brush_drawing:
            return

        z = self.patch_z
        add_mode = (button_mode == 1)
        self.apply_brush(z, x, y, add_mode)
        self.update_metric()
        self.refresh_patch_views()
        self.update_texts()

    def on_editor_release(self):
        self.is_brush_drawing = False

    def clear_polygon_points(self):
        self.poly_points = []
        self.refresh_patch_views()
        self.update_texts()

    def apply_polygon_by_mode(self):
        if self.edited_mask is None:
            QMessageBox.warning(self, "提示", "请先完成模型预测。")
            return

        if not self.is_polygon_mode():
            QMessageBox.information(self, "提示", "当前编辑模式不是多边形模式。")
            return

        if len(self.poly_points) < 3:
            QMessageBox.warning(self, "提示", "至少需要 3 个点才能形成闭合区域。")
            return

        z = max(0, min(self.edited_mask.shape[0] - 1, self.patch_z))
        h, w = self.edited_mask[z].shape
        poly_mask = polygon_to_mask(h, w, self.poly_points)

        self.push_undo_state()

        if self.get_edit_mode() == "多边形补":
            self.edited_mask[z][poly_mask > 0] = 1.0
        else:
            self.edited_mask[z][poly_mask > 0] = 0.0

        self.poly_points = []
        self.update_metric()
        self.refresh_patch_views()
        self.update_texts()

    def apply_current_patch_to_whole_ct(self):
        if self.ct_volume is None or self.edited_mask is None or self.current_center_zyx is None:
            QMessageBox.warning(self, "提示", "请先完成当前结节分割和修正。")
            return

        cz, cy, cx = self.current_center_zyx.tolist()
        self.whole_ct_mask = paste_mask_back_3d(
            self.whole_ct_mask,
            (self.edited_mask > 0.5).astype(np.uint8),
            cz, cy, cx
        )

        self.refresh_ct_views()
        self.update_texts()
        QMessageBox.information(self, "提示", "当前结节修正结果已回贴到整CT。")

    def clear_whole_ct_mask(self):
        if self.ct_volume is None:
            QMessageBox.warning(self, "提示", "请先打开CT。")
            return
        self.whole_ct_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
        self.refresh_ct_views()
        self.update_texts()
        QMessageBox.information(self, "提示", "整CT累计 mask 已清空。")

    def update_texts(self):
        file_text = f"当前CT：{Path(self.ct_file).name if self.ct_file else '未加载'}"
        if self.series_uid is not None:
            file_text += f"\nSeriesUID：{self.series_uid}"
        if self.source_type is not None:
            file_text += f"\n来源类型：{self.source_type}"
        if self.ct_folder is not None:
            file_text += f"\n当前目录：{self.ct_folder}"

        if self.ct_volume is not None:
            z, y, x = self.ct_volume.shape
            file_text += f"\n体积大小(Z,Y,X)：({z}, {y}, {x})"

        if self.ct_spacing_xyz is not None:
            sx, sy, sz = self.ct_spacing_xyz.tolist()
            file_text += f"\nSpacing(x,y,z)：({sx:.3f}, {sy:.3f}, {sz:.3f})"

        file_text += f"\nPatch大小：{self.patch_size_spin.value()} | 阈值：{self.threshold_spin.value():.2f}"
        file_text += f"\n窗位/窗宽：WL={self.window_level:.0f}, WW={self.window_width:.0f}"
        file_text += (
            f"\nCT透明度(肺/结节)：{self.lung_alpha:.2f}/{self.nodule_alpha:.2f}"
            f"\nPatch透明度(GT/Pred)：{self.patch_gt_alpha:.2f}/{self.patch_pred_alpha:.2f}"
        )
        file_text += (
            f"\nCC后处理：{'开启' if config.POSTPROCESS_ENABLE else '关闭'}"
            f" | method={config.POSTPROCESS_METHOD}"
            f" | conn={config.POSTPROCESS_CONNECTIVITY}"
            f" | min_size={config.POSTPROCESS_MIN_SIZE}"
        )
        self.file_label.setText(file_text)

        total_ct = len(self.ct_file_list)
        if total_ct > 0 and self.current_ct_index >= 0:
            self.progress_label.setText(f"CT进度: 第 {self.current_ct_index + 1} / {total_ct} 个")
        else:
            self.progress_label.setText("CT进度: -")

        if self.manual_mode:
            self.mode_label.setText("当前中心来源: 手动选择")
        else:
            self.mode_label.setText("当前中心来源: 标注结节")

        if self.whole_ct_mask is not None:
            fg_voxels = int(self.whole_ct_mask.sum())
            if self.lung_mask is not None:
                lung_voxels = int(self.lung_mask.sum())
                self.whole_mask_label.setText(f"整CT mask 体素数: {fg_voxels} | lung体素数: {lung_voxels}")
            else:
                self.whole_mask_label.setText(f"整CT mask 体素数: {fg_voxels}")
        else:
            self.whole_mask_label.setText("整CT mask 体素数: -")

        self.poly_label.setText(f"当前多边形点数: {len(self.poly_points)} | 当前编辑模式: {self.get_edit_mode()}")

        if self.current_center_xyz is not None:
            ix, iy, iz = self.current_center_xyz.tolist()
            dia_text = f"{self.current_diameter_mm:.2f}" if self.current_diameter_mm is not None else "-"
            extra = ""
            if self.edited_mask is not None:
                fg_patch = int(self.edited_mask.sum())
                extra = f"\nPatch前景体素数: {fg_patch}"
            self.center_label.setText(
                f"当前中心 voxel(x,y,z)=({ix}, {iy}, {iz}) | diameter={dia_text} mm | nodule_index={self.current_nodule_index}{extra}"
            )
        else:
            self.center_label.setText("当前中心: 尚未设置")

        patch_z_value = self.patch_z if self.patch is not None else -1
        self.slice_label.setText(
            f"patch_z: {patch_z_value} | axial_z: {self.axial_z} | coronal_y: {self.coronal_y} | sagittal_x: {self.sagittal_x}"
        )

        self.info_label.setText(
            "快捷操作：滚轮切层；Ctrl+滚轮缩放。"
            "左键点击CT图可指定中心；Pred图刷子模式下左键补前景、右键擦除；"
            "多边形模式下左键加点、右键删除最后一个点。绿色=肺区域，红色=结节mask。"
        )

    def safe_filename(name: str) -> str:
        text = str(name)
        for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            text = text.replace(ch, "_")
        return text.strip()


    def save_current_pred_png(self):
        if self.patch is None or self.edited_mask is None:
            QMessageBox.warning(self, "提示", "请先完成分割。")
            return

        z = self.patch_z
        overlay = overlay_mask_on_hu(
            self.patch[z],
            self.edited_mask[z],
            color="red",
            alpha=self.patch_pred_alpha,
            wl=self.window_level,
            ww=self.window_width
        )

        base_name = safe_filename(Path(self.ct_file).name) if self.ct_file else "unknown"
        thr = float(self.threshold_spin.value())
        out_path = SAVE_DIR / f"{base_name}_n{self.current_nodule_index}_z{z}_thr{thr:.2f}_edited_pred.png"

        h, w, _ = overlay.shape
        qimg = QImage(overlay.data, w, h, w * 3, QImage.Format_RGB888)
        ok = qimg.save(str(out_path))

        if ok:
            QMessageBox.information(self, "保存成功", f"已保存到：\n{out_path}")
        else:
            QMessageBox.critical(self, "保存失败", "PNG 保存失败。")

    def save_current_mask_npy(self):
        if self.edited_mask is None:
            QMessageBox.warning(self, "提示", "请先完成分割。")
            return

        base_name = safe_filename(Path(self.ct_file).name) if self.ct_file else "unknown"
        thr = float(self.threshold_spin.value())
        out_path = SAVE_DIR / f"{base_name}_n{self.current_nodule_index}_thr{thr:.2f}_edited_patch_mask.npy"

        np.save(out_path, self.edited_mask.astype(np.uint8))
        QMessageBox.information(self, "保存成功", f"修正后 Patch 3D Mask 已保存到：\n{out_path}")

    def save_current_whole_ct_mask_mhd(self):
        if self.ct_volume is None or self.whole_ct_mask is None:
            QMessageBox.warning(self, "提示", "请先打开CT。")
            return

        if int(self.whole_ct_mask.sum()) == 0:
            QMessageBox.warning(self, "提示", "当前整CT累计 mask 为空。请先点击“把当前修正结果回贴到整CT”。")
            return

        try:
            base_name = safe_filename(Path(self.ct_file).name) if self.ct_file else "unknown"
            out_mhd = SAVE_DIR / f"{base_name}_whole_ct_mask_accumulated.mhd"
            out_npy = SAVE_DIR / f"{base_name}_whole_ct_mask_accumulated.npy"

            np.save(out_npy, self.whole_ct_mask.astype(np.uint8))

            mask_img = sitk.GetImageFromArray(self.whole_ct_mask.astype(np.uint8))
            mask_img.SetOrigin(tuple(self.ct_origin_xyz.tolist()))
            mask_img.SetSpacing(tuple(self.ct_spacing_xyz.tolist()))
            if self.ct_direction is not None:
                mask_img.SetDirection(self.ct_direction)

            sitk.WriteImage(mask_img, str(out_mhd))

            QMessageBox.information(
                self,
                "保存成功",
                f"累计后的整CT mask 已保存：\n{out_mhd}\n\n同时也保存了：\n{out_npy}"
            )

        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))


def build_app_style(base_font: int):
    btn_h = max(34, base_font * 3)
    input_h = max(30, base_font * 2 + 8)
    radius = 8

    return f"""
        QWidget {{
            font-size: {base_font}px;
            color: #f0f0f0;
            background: #1f2329;
        }}

        QMainWindow {{
            background: #1f2329;
        }}

        QLabel {{
            font-size: {base_font}px;
            color: #f0f0f0;
            background: transparent;
        }}

        QPushButton {{
            font-size: {base_font}px;
            min-height: {btn_h}px;
            padding: 4px 8px;
            border: 1px solid #5c6470;
            border-radius: {radius}px;
            background: #31363f;
            color: #ffffff;
        }}

        QPushButton:hover {{
            background: #3a404a;
        }}

        QPushButton:pressed {{
            background: #252a31;
        }}

        QComboBox, QSpinBox, QDoubleSpinBox {{
            font-size: {base_font}px;
            min-height: {input_h}px;
            padding: 2px 6px;
            border: 1px solid #5c6470;
            border-radius: 6px;
            background: #2a2f36;
            color: #ffffff;
        }}

        QFrame#InfoCard {{
            border: 1px solid #4f5560;
            border-radius: 10px;
            background: #262b33;
        }}
    """


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    screen = QApplication.primaryScreen().availableGeometry()
    if screen.width() <= 1366:
        base_font = 12
    elif screen.width() <= 1600:
        base_font = 13
    else:
        base_font = 15

    app.setStyleSheet(build_app_style(base_font))

    font = QFont()
    font.setPointSize(base_font)
    app.setFont(font)

    window = CTDirectInferFolderApp()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("GUI 启动失败：%s", e)
        raise