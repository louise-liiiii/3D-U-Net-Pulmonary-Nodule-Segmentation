import numpy as np
import SimpleITK as sitk
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from .geom_utils import safe_crop_3d, make_sphere_mask

from .config import (
    LUNA_SUBSET0_DIR,
    LUNA_ANN_CSV,
    BUILD_OUT_DIR,
    PATCH_SIZE,
    MIN_DIAMETER_MM,
    MAX_DIAMETER_MM,
    TARGET_SPACING_XYZ,
    validate_raw_dir,
)

logger = logging.getLogger(__name__)


def resample_sitk_image(image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = np.array(image.GetSpacing(), dtype=np.float32)
    original_size = np.array(image.GetSize(), dtype=np.int32)

    out_spacing = np.array(out_spacing, dtype=np.float32)
    out_size = np.maximum(
        np.round(original_size * (original_spacing / out_spacing)).astype(np.int32),
        1
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetSize(tuple(out_size.tolist()))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def make_npz_name(seriesuid: str, nodule_idx: int, d_mm: float) -> str:
    return f"{seriesuid}_n{nodule_idx}_d{d_mm:.2f}_p{PATCH_SIZE}.npz"


def run_build_subset0():
    validate_raw_dir()

    subset_dir = Path(LUNA_SUBSET0_DIR)
    ann_path = Path(LUNA_ANN_CSV)
    out_dir = Path(BUILD_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not subset_dir.exists():
        raise RuntimeError(f"subset dir not found: {subset_dir}")
    if not ann_path.exists():
        raise RuntimeError(f"annotations.csv not found: {ann_path}")

    df = pd.read_csv(ann_path)
    mhd_files = sorted(list(subset_dir.glob("*.mhd")))

    logger.info("Found mhd files: %d", len(mhd_files))
    logger.info("Build out dir: %s", out_dir)
    logger.info("Filter diameter: %.1f–%.1f mm", MIN_DIAMETER_MM, MAX_DIAMETER_MM)
    logger.info("Patch size: %d", PATCH_SIZE)

    total_ct = 0
    saved_npz = 0
    meta_rows = []
    half = PATCH_SIZE // 2

    for mhd_path in tqdm(mhd_files, desc="BUILD subset0"):
        total_ct += 1
        seriesuid = mhd_path.stem

        df_case = df[df["seriesuid"] == seriesuid]
        if len(df_case) == 0:
            continue

        df_small = df_case[
            (df_case["diameter_mm"] >= MIN_DIAMETER_MM) &
            (df_case["diameter_mm"] <= MAX_DIAMETER_MM)
        ]
        if len(df_small) == 0:
            continue

        try:
            img = sitk.ReadImage(str(mhd_path))
            img = resample_sitk_image(img, out_spacing=TARGET_SPACING_XYZ, is_label=False)

            vol = sitk.GetArrayFromImage(img).astype(np.int16)
            origin = np.array(img.GetOrigin(), dtype=np.float32)
            spacing = np.array(img.GetSpacing(), dtype=np.float32)
        except Exception as e:
            logger.warning("Skip bad CT file: %s | error=%s", mhd_path, e)
            continue

        for i, row in df_small.reset_index(drop=True).iterrows():
            d_mm = float(row["diameter_mm"])
            x_world = float(row["coordX"])
            y_world = float(row["coordY"])
            z_world = float(row["coordZ"])

            ix, iy, iz = img.TransformPhysicalPointToIndex((x_world, y_world, z_world))
            ix, iy, iz = int(ix), int(iy), int(iz)

            Z, Y, X = vol.shape
            if not (0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X):
                logger.warning(
                    "Skip nodule out of bounds: %s | world=(%.3f, %.3f, %.3f) | voxel=(%d, %d, %d) | shape=(%d, %d, %d)",
                    seriesuid, x_world, y_world, z_world, ix, iy, iz, X, Y, Z
                )
                continue

            patch = safe_crop_3d(vol, iz, iy, ix, patch=PATCH_SIZE)
            spacing_zyx = spacing[::-1].copy().astype(np.float32)
            mask = make_sphere_mask(PATCH_SIZE, spacing_zyx, d_mm).astype(np.uint8)

            npz_name = make_npz_name(seriesuid, i, d_mm)
            out_path = out_dir / npz_name
            center_in_patch_zyx = (half, half, half)

            np.savez_compressed(
                out_path,
                patch=patch.astype(np.int16),
                mask=mask,
                center_xyz=np.array([ix, iy, iz], dtype=np.int32),
                center_in_patch_zyx=np.array(center_in_patch_zyx, dtype=np.int32),
                diameter_mm=np.array([d_mm], dtype=np.float32),
                spacing=spacing.astype(np.float32),
                origin=origin.astype(np.float32),
                world_xyz=np.array([x_world, y_world, z_world], dtype=np.float32),
                seriesuid=np.array([seriesuid]),
            )

            meta_rows.append({
                "npz_name": npz_name,
                "mhd_path": str(mhd_path),
                "seriesuid": seriesuid,
                "nodule_idx": int(i),
                "diameter_mm": float(d_mm),
                "coordX": x_world,
                "coordY": y_world,
                "coordZ": z_world,
                "ix": int(ix),
                "iy": int(iy),
                "iz": int(iz),
                "patch_size": int(PATCH_SIZE),
            })

            saved_npz += 1

    meta_csv = out_dir / "metadata.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False, encoding="utf-8-sig")

    summary = [
        f"Total CT scanned: {total_ct}",
        f"Saved npz: {saved_npz}",
        f"Output dir: {out_dir}",
        f"Metadata csv: {meta_csv}",
    ]
    (out_dir / "_build_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    logger.info("\n".join(summary))