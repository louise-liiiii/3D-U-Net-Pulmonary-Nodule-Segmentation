# 3D U-Net for Pulmonary Nodule CT Segmentation

## Overview
This project implements a 3D U-Net with CBAM attention for automatic segmentation of small pulmonary nodules (3–10 mm) in CT images. It was developed as my undergraduate graduation thesis and further extended into a PyQt-based interactive segmentation system.

## Results
| Method | Dice | IoU | HD95 (mm) |
|---|---:|---:|---:|
| 3D U-Net + Dice Loss | 0.5712 | 0.4408 | 1.8421 |
| 3D U-Net + Dice + Focal Loss | 0.6385 | 0.4994 | 1.8543 |
| **3D U-Net + CBAM + Dice Loss** | **0.7228** | **0.6006** | **1.4336** |
| 3D U-Net + CBAM + Dice + Focal Loss | 0.5149 | 0.4234 | 1.4704 |

Best model: **3D U-Net + CBAM + Dice Loss**  
Dice = 0.7228, IoU = 0.6006, HD95 = 1.4336 mm.

## Method
- Dataset: LUNA16, 84 nodules with diameter 3–10 mm
- Input: 64×64×64 3D patches centered on nodules
- Preprocessing: resampling to 1×1×1 mm, normalization, data augmentation
- Network: 3D U-Net with CBAM in skip connections
- Loss: Dice Loss, Dice + Focal Loss compared
- Post-processing: threshold search on validation set + 3D connected-component analysis
- System: PyQt GUI with tri-planar visualization, threshold adjustment, brush and polygon tools

## Tech Stack
- Python, PyTorch
- PyQt5
- SimpleITK, NumPy, cc3d
- MATLAB (early image processing course project)

## Repository Structure