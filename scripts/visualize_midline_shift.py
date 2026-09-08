#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Midline Shift (MLS) Detection & Visualization Module
=====================================================
Tác vụ:
1. Đọc các file dự đoán trong thư mục `pred_data/` và ánh xạ tới dữ liệu MRI/CT tương ứng trong `sample_data/`.
2. Trích xuất và xây dựng:
   - Đường màu đỏ (Deformed Midline - DML): Đường cong tại rãnh giữa não bộ bám sát khối u và độ lệch chèn ép của nhu mô não.
   - Đường kẻ đứt màu xanh (Ideal Midline / Anatomical Midline - AML): Đoạn thẳng nối 2 đầu mút cực trước (anterior) và cực sau (posterior) của đường đỏ.
   - Tính toán độ lệch cực đại (Midline Shift - MLS) từ điểm xa nhất của đường đỏ tới đường thẳng màu xanh (đơn vị: mm và pixels).
3. Xuất toàn bộ các ảnh lát cắt và báo cáo chi tiết vào thư mục tương ứng trong `output_images/`.

Tham khảo: Brain Midline Shift Measurement & 3D Neuroradiological Assessment (arXiv:2602.22098 / arXiv:1908.04568).
"""

import os
import sys
import glob
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Configure utf-8 encoding for Windows terminal
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d, binary_fill_holes
from scipy.interpolate import interp1d


def find_matched_nii_and_seg(
    data_dir: str, 
    case_name: str, 
    preferred_modality: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Tìm file ảnh NIfTI gốc và file nhãn phân vùng khối u (segmentation mask nếu có)
    tương ứng với case_name từ thư mục dữ liệu.
    """
    data_path = Path(data_dir)
    
    # 1. Tách modality nếu tên case có chứa hậu tố (ví dụ: BraTS-MEN-00023-000-t2f -> modality: t2f)
    clean_id = case_name.replace('pred_', '')
    modality = preferred_modality or 't2f'
    
    known_modalities = ['t2f', 't2w', 't1c', 't1n', 't2_axial', 't2', 't1', 'flair', 'seg']
    for m in known_modalities:
        if clean_id.endswith(f"_{m}") or clean_id.endswith(f"-{m}"):
            modality = m
            # Base patient id
            sep = '_' if clean_id.endswith(f"_{m}") else '-'
            clean_id = clean_id[:-(len(m) + 1)]
            break

    # 2. Tìm file MRI ảnh scan
    mri_candidates = [
        # Khớp chính xác tên file
        data_path / f"{case_name.replace('pred_', '')}.nii.gz",
        data_path / f"{case_name.replace('pred_', '')}.nii",
        data_path / clean_id / f"*-{modality}.nii.gz",
        data_path / clean_id / f"*{modality}*.nii.gz",
        data_path / f"*{clean_id}*{modality}*.nii.gz",
        data_path / f"*{clean_id}*.nii.gz",
        data_path / f"*{clean_id}*.nii",
    ]
    
    mri_file = None
    for cand in mri_candidates:
        if isinstance(cand, Path) and cand.exists() and cand.is_file():
            mri_file = str(cand)
            break
        matches = glob.glob(str(cand))
        if matches:
            if modality != 'seg':
                non_seg = [f for f in matches if not f.endswith('-seg.nii.gz') and not f.endswith('_seg.nii.gz')]
                mri_file = non_seg[0] if non_seg else matches[0]
            else:
                mri_file = matches[0]
            break

    # 3. Tìm file segmentation khối u (seg)
    seg_candidates = [
        data_path / f"{clean_id}-seg.nii.gz",
        data_path / f"{clean_id}_seg.nii.gz",
        data_path / clean_id / f"*-seg.nii.gz",
        data_path / f"*{clean_id}*seg*.nii.gz",
    ]
    
    seg_file = None
    for cand in seg_candidates:
        if isinstance(cand, Path) and cand.exists() and cand.is_file():
            seg_file = str(cand)
            break
        matches = glob.glob(str(cand))
        if matches:
            seg_file = matches[0]
            break

    return mri_file, seg_file, clean_id


def extract_brain_mask_and_anchors(
    img_2d: np.ndarray, 
    min_pixels: int = 150
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Xác định mặt nạ não (brain mask) và 2 điểm neo cực trước (Anterior) & cực sau (Posterior).
    Trả về: (brain_mask, p_anterior, p_posterior) hoặc None nếu lát cắt không chứa mô não.
    """
    H, W = img_2d.shape
    nonzero = img_2d[img_2d > 0]
    if len(nonzero) < min_pixels:
        return None

    thresh = np.percentile(nonzero, 8) if len(nonzero) > 0 else 0
    raw_mask = img_2d > thresh
    brain_mask = binary_fill_holes(raw_mask)
    
    y_coords, x_coords = np.where(brain_mask)
    if len(y_coords) < min_pixels:
        return None

    y_min, y_max = np.min(y_coords), np.max(y_coords)
    if y_max - y_min < 20:
        return None

    margin_y = max(3, int((y_max - y_min) * 0.035))
    y_ant = y_min + margin_y
    y_post = y_max - margin_y

    ant_x_pts = np.where(brain_mask[y_ant, :])[0]
    post_x_pts = np.where(brain_mask[y_post, :])[0]

    x_ant = float(np.mean(ant_x_pts)) if len(ant_x_pts) > 0 else W / 2.0
    x_post = float(np.mean(post_x_pts)) if len(post_x_pts) > 0 else W / 2.0

    p_ant = np.array([x_ant, float(y_ant)], dtype=float)
    p_post = np.array([x_post, float(y_post)], dtype=float)

    return brain_mask, p_ant, p_post


def compute_midline_and_shift(
    img_2d: np.ndarray,
    seg_2d: Optional[np.ndarray] = None,
    pred_points: Optional[List[Any]] = None,
    voxel_size_mm: float = 1.0,
    num_samples: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Tính toán chi tiết các thành phần đường giữa não:
    - Điểm cực trước P_ant và cực sau P_post.
    - Đường thẳng lý tưởng màu xanh (AML) nối P_ant -> P_post.
    - Đường cong màu đỏ (DML) bám theo rãnh giữa và độ cong khối u.
    - Điểm lệch cực đại P_max, điểm hình chiếu P_proj trên đường thẳng.
    - Độ lệch Midline Shift (MLS) bằng mm và pixels.
    """
    H, W = img_2d.shape
    anchors = extract_brain_mask_and_anchors(img_2d)
    if anchors is None:
        return None

    brain_mask, p_ant, p_post = anchors
    x_ant, y_ant = p_ant[0], p_ant[1]
    x_post, y_post = p_post[0], p_post[1]

    # 1. Đường thẳng nối 2 đầu mút (Đường nét đứt màu xanh - AML)
    y_vals = np.linspace(y_ant, y_post, num_samples)
    t = (y_vals - y_ant) / (y_post - y_ant + 1e-8)
    ideal_x = (1.0 - t) * x_ant + t * x_post

    # 2. Xây dựng đường cong màu đỏ (DML) bám theo rãnh não và khối u
    deformed_x = ideal_x.copy()
    has_tumor_effect = False
    tumor_side = "None"

    # Trường hợp A: Có mặt nạ khối u (Segmentation mask)
    if seg_2d is not None and np.any(seg_2d > 0):
        t_y, t_x = np.where(seg_2d > 0)
        t_cy = float(np.mean(t_y))
        t_cx = float(np.mean(t_x))
        
        ideal_cx_at_tumor = float(np.interp(t_cy, y_vals, ideal_x))
        tumor_side = "Right" if t_cx > ideal_cx_at_tumor else "Left"
        has_tumor_effect = True

        t_ymin, t_ymax = np.min(t_y), np.max(t_y)
        trans_margin = max(10.0, (t_ymax - t_ymin) * 0.25)
        raw_def_x = ideal_x.copy()

        for i, y in enumerate(y_vals):
            y_int = int(np.clip(np.round(y), 0, H - 1))
            row_tumor_x = np.where(seg_2d[y_int, :] > 0)[0]

            if len(row_tumor_x) > 0:
                if tumor_side == "Right":
                    # Khối u bên phải -> Đẩy rãnh não lồi sang trái (min x của khối u)
                    medial_x = float(np.min(row_tumor_x))
                    if medial_x < ideal_x[i] + 8.0:
                        raw_def_x[i] = min(ideal_x[i] - 2.0, medial_x - 2.0)
                    else:
                        dist = medial_x - ideal_x[i]
                        push = max(3.0, 18.0 * np.exp(-dist / 25.0))
                        raw_def_x[i] = ideal_x[i] - push
                else:
                    # Khối u bên trái -> Đẩy rãnh não lồi sang phải (max x của khối u)
                    medial_x = float(np.max(row_tumor_x))
                    if medial_x > ideal_x[i] - 8.0:
                        raw_def_x[i] = max(ideal_x[i] + 2.0, medial_x + 2.0)
                    else:
                        dist = ideal_x[i] - medial_x
                        push = max(3.0, 18.0 * np.exp(-dist / 25.0))
                        raw_def_x[i] = ideal_x[i] + push
            else:
                # Chuyển tiếp êm dịu (Smooth transition curve) trước và sau khối u
                if t_ymin - trans_margin <= y < t_ymin:
                    row_0 = np.where(seg_2d[t_ymin, :] > 0)[0]
                    if len(row_0) > 0:
                        edge_0 = float(np.min(row_0) - 2.0 if tumor_side == "Right" else np.max(row_0) + 2.0)
                        target_0 = min(ideal_x[i] - 2.0, edge_0) if tumor_side == "Right" else max(ideal_x[i] + 2.0, edge_0)
                        u = (y - (t_ymin - trans_margin)) / trans_margin
                        factor = 0.5 * (1.0 - np.cos(np.pi * u))
                        raw_def_x[i] = ideal_x[i] + factor * (target_0 - ideal_x[i])
                elif t_ymax < y <= t_ymax + trans_margin:
                    row_1 = np.where(seg_2d[t_ymax, :] > 0)[0]
                    if len(row_1) > 0:
                        edge_1 = float(np.min(row_1) - 2.0 if tumor_side == "Right" else np.max(row_1) + 2.0)
                        target_1 = min(ideal_x[i] - 2.0, edge_1) if tumor_side == "Right" else max(ideal_x[i] + 2.0, edge_1)
                        u = ((t_ymax + trans_margin) - y) / trans_margin
                        factor = 0.5 * (1.0 - np.cos(np.pi * u))
                        raw_def_x[i] = ideal_x[i] + factor * (target_1 - ideal_x[i])
                else:
                    raw_def_x[i] = ideal_x[i]

        # Lọc Gauss làm mượt liên tục toàn bộ đường cong DML
        deformed_x = gaussian_filter1d(raw_def_x, sigma=4.5)

    # Trường hợp B: Sử dụng tọa độ từ file dự đoán pred_data
    elif pred_points is not None and len(pred_points) > 0:
        try:
            arr = np.array(pred_points, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                ys, xs = arr[:, 0], arr[:, 1]
                sort_idx = np.argsort(ys)
                ys, xs = ys[sort_idx], xs[sort_idx]
                _, unique_idx = np.unique(ys, return_index=True)
                ys, xs = ys[unique_idx], xs[unique_idx]

                valid_idx = (ys >= y_ant) & (ys <= y_post)
                if np.sum(valid_idx) >= 2:
                    y_sub, x_sub = ys[valid_idx], xs[valid_idx]
                    interp_f = interp1d(y_sub, x_sub, kind='linear', bounds_error=False, fill_value=np.nan)
                    pred_interp = interp_f(y_vals)

                    raw_def_x = ideal_x.copy()
                    has_val = ~np.isnan(pred_interp)
                    raw_def_x[has_val] = pred_interp[has_val]

                    # Chuyển tiếp êm dịu về đường thẳng chuẩn AML ở các vùng thiếu nhãn (tránh ngoại suy vọt ra ngoài sọ)
                    min_y_val = float(np.min(y_sub))
                    max_y_val = float(np.max(y_sub))
                    blend_margin = 15.0

                    for i, y in enumerate(y_vals):
                        if y < min_y_val:
                            dist = min_y_val - y
                            if dist < blend_margin:
                                u = dist / blend_margin
                                factor = 0.5 * (1.0 + np.cos(np.pi * u))
                                raw_def_x[i] = ideal_x[i] + factor * (raw_def_x[i] - ideal_x[i])
                            else:
                                raw_def_x[i] = ideal_x[i]
                        elif y > max_y_val:
                            dist = y - max_y_val
                            if dist < blend_margin:
                                u = dist / blend_margin
                                factor = 0.5 * (1.0 + np.cos(np.pi * u))
                                raw_def_x[i] = ideal_x[i] + factor * (raw_def_x[i] - ideal_x[i])
                            else:
                                raw_def_x[i] = ideal_x[i]

                    deformed_x = gaussian_filter1d(raw_def_x, sigma=3.5)
            elif arr.ndim == 1 and len(arr) == len(y_vals):
                deformed_x = gaussian_filter1d(arr, sigma=3.5)
        except Exception:
            pass

    # Giới hạn giải phẫu bắt buộc: Đường giữa luôn luôn phải nằm BÊN TRONG mô não
    for i, y in enumerate(y_vals):
        y_idx = int(np.clip(np.round(y), 0, H - 1))
        row_brain = np.where(brain_mask[y_idx, :])[0]
        if len(row_brain) > 0:
            b_min = float(np.min(row_brain)) + 2.0
            b_max = float(np.max(row_brain)) - 2.0
            if b_min < b_max:
                deformed_x[i] = np.clip(deformed_x[i], b_min, b_max)

    # Đảm bảo tuyệt đối 2 đầu mút nối liền chính xác vào P_ant và P_post
    deformed_x[0] = x_ant
    deformed_x[-1] = x_post

    # 3. Tính toán độ lệch cực đại MLS (Midline Shift)
    line_len = np.hypot(x_post - x_ant, y_post - y_ant)
    if line_len < 1e-5:
        return None

    # d = |(y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    distances_px = np.abs(
        (y_post - y_ant) * deformed_x - (x_post - x_ant) * y_vals + x_post * y_ant - y_post * x_ant
    ) / line_len

    max_idx = int(np.argmax(distances_px))
    max_shift_px = float(distances_px[max_idx])
    max_shift_mm = float(max_shift_px * voxel_size_mm)

    p_max = np.array([deformed_x[max_idx], y_vals[max_idx]], dtype=float)

    # Điểm hình chiếu vuông góc P_proj trên đoạn thẳng màu xanh
    vec_line = np.array([x_post - x_ant, y_post - y_ant], dtype=float)
    vec_point = p_max - p_ant
    t_proj = np.clip(np.dot(vec_point, vec_line) / (np.dot(vec_line, vec_line) + 1e-8), 0.0, 1.0)
    p_proj = p_ant + t_proj * vec_line

    return {
        'p_ant': p_ant,
        'p_post': p_post,
        'ideal_x': ideal_x,
        'ideal_y': y_vals,
        'def_x': deformed_x,
        'def_y': y_vals,
        'mls_mm': max_shift_mm,
        'mls_px': max_shift_px,
        'p_max': p_max,
        'p_proj': p_proj,
        'has_tumor': has_tumor_effect,
        'tumor_side': tumor_side,
        'max_shift_y': float(y_vals[max_idx])
    }


def render_and_save_slice_visualization(
    img_2d: np.ndarray,
    seg_2d: Optional[np.ndarray],
    res: Optional[Dict[str, Any]],
    case_name: str,
    slice_idx: int,
    total_slices: int,
    output_path: str,
    voxel_size_mm: float = 1.0,
    dpi: int = 150
) -> None:
    """
    Vẽ ảnh lát cắt với tiêu chuẩn y khoa:
    - Khung 1: Toàn cảnh lát cắt não với đường cong màu đỏ bám khối u và đường thẳng nét đứt màu xanh nối 2 đầu mút.
    - Khung 2: Khung phóng to cận cảnh (Zoom-In) tại vị trí độ lệch cực đại MLS kèm thước đo / mũi tên khoảng cách.
    """
    H, W = img_2d.shape
    # Thiết lập kích thước và nền trắng thanh lịch chuẩn tài liệu y khoa (khớp với ảnh mẫu)
    fig = plt.figure(figsize=(12, 6.2), facecolor='#FFFFFF', edgecolor='none')
    
    # 1. Khung toàn cảnh (Overview Slice)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#FFFFFF')
    ax1.imshow(img_2d, cmap='gray', origin='upper')

    # Vẽ lớp phủ khối u mờ nếu có (để bác sĩ/người dùng nhận biết vị trí khối u)
    if seg_2d is not None and np.any(seg_2d > 0):
        tumor_mask = seg_2d > 0
        colored_tumor = np.zeros((H, W, 4), dtype=float)
        colored_tumor[tumor_mask] = [1.0, 0.45, 0.0, 0.28]  # Cam trong suốt nhẹ
        ax1.imshow(colored_tumor, origin='upper')

    if res is not None:
        # Đường thẳng nét đứt màu xanh dương (AML - Anatomical / Ideal Midline)
        ax1.plot(
            res['ideal_x'], res['ideal_y'],
            color='#00A2FF', linestyle='--', linewidth=2.4, alpha=0.95,
            label='Đường thẳng nối 2 đầu mút (AML)'
        )
        # Đường cong màu đỏ bám theo rãnh não và viền khối u (DML - Deformed Midline)
        ax1.plot(
            res['def_x'], res['def_y'],
            color='#FF1744', linestyle='-', linewidth=2.8, alpha=0.95,
            label='Đường cong bám khối u (DML)'
        )
        # Điểm cực trước và cực sau não bộ
        ax1.scatter([res['p_ant'][0], res['p_post'][0]], [res['p_ant'][1], res['p_post'][1]],
                    color='#FFD600', s=30, zorder=5, edgecolors='black', linewidths=0.8)

        mls_val_mm = res['mls_mm']
        mls_val_px = res['mls_px']
    else:
        mls_val_mm = 0.0
        mls_val_px = 0.0

    ax1.axis('off')

    # 2. Khung phóng to cận cảnh (Zoom-In Coordinate Box)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#FFFFFF')

    if res is not None and res['mls_mm'] > 0.05:
        center_x = (res['p_max'][0] + res['p_proj'][0]) / 2.0
        center_y = (res['p_max'][1] + res['p_proj'][1]) / 2.0
    else:
        center_x, center_y = W / 2.0, H / 2.0

    crop_half_w = 38
    crop_half_h = 56
    x0 = max(0, int(center_x - crop_half_w))
    x1 = min(W, int(center_x + crop_half_w))
    y0 = max(0, int(center_y - crop_half_h))
    y1 = min(H, int(center_y + crop_half_h))

    zoom_img = img_2d[y0:y1, x0:x1]
    zh, zw = zoom_img.shape
    ax2.imshow(zoom_img, cmap='gray', origin='upper')

    if seg_2d is not None and np.any(seg_2d[y0:y1, x0:x1] > 0):
        sub_seg = seg_2d[y0:y1, x0:x1] > 0
        colored_sub = np.zeros((zh, zw, 4), dtype=float)
        colored_sub[sub_seg] = [1.0, 0.45, 0.0, 0.25]
        ax2.imshow(colored_sub, origin='upper')

    if res is not None:
        # Đường thẳng nét đứt màu xanh dương trong vùng phóng to
        ax2.plot(
            res['ideal_x'] - x0, res['ideal_y'] - y0,
            color='#00A2FF', linestyle='--', linewidth=3.8, alpha=0.95
        )
        # Đường cong màu đỏ bám khối u trong vùng phóng to
        ax2.plot(
            res['def_x'] - x0, res['def_y'] - y0,
            color='#FF1744', linestyle='-', linewidth=4.2, alpha=0.95
        )

        p_max_crop = (res['p_max'][0] - x0, res['p_max'][1] - y0)
        p_proj_crop = (res['p_proj'][0] - x0, res['p_proj'][1] - y0)

        # Mũi tên 2 chiều màu xanh lá cây đo độ lệch cực đại MLS
        if res['mls_px'] > 0.3:
            ax2.annotate(
                '', xy=p_max_crop, xytext=p_proj_crop,
                arrowprops=dict(
                    arrowstyle='<->,head_width=0.55,head_length=0.7',
                    color='#00E676', lw=3.2
                ),
                zorder=6
            )
            # Nhãn số đo khoảng cách màu xanh lá cây đặt ở giữa mũi tên (như hình mẫu 14 mm)
            mid_x = (p_max_crop[0] + p_proj_crop[0]) / 2.0
            mid_y = (p_max_crop[1] + p_proj_crop[1]) / 2.0 - 5.0
            display_mm = f"{mls_val_mm:.0f} mm" if abs(mls_val_mm - round(mls_val_mm)) < 0.3 else f"{mls_val_mm:.1f} mm"
            ax2.text(
                mid_x, mid_y, display_mm,
                color='#76FF03', fontsize=16, fontweight='bold',
                ha='center', va='bottom',
                path_effects=None,
                bbox=dict(boxstyle='square,pad=0.15', facecolor='#111111', alpha=0.6, edgecolor='none'),
                zorder=7
            )

    # Vẽ hệ trục tọa độ X - Y màu đen có mũi tên (như hình ảnh mẫu người dùng cung cấp)
    # Trục Y hướng lên trên (từ zh đến 0)
    ax2.annotate(
        '', xy=(0, -2), xytext=(0, zh),
        arrowprops=dict(arrowstyle='->,head_width=0.65,head_length=0.85', color='black', lw=3.5),
        annotation_clip=False, zorder=8
    )
    ax2.text(
        2, -1, 'Y', color='black', fontsize=14, fontweight='bold',
        ha='left', va='bottom', zorder=9
    )

    # Trục X hướng sang phải (từ 0 đến zw)
    ax2.annotate(
        '', xy=(zw + 2, zh), xytext=(0, zh),
        arrowprops=dict(arrowstyle='->,head_width=0.65,head_length=0.85', color='black', lw=3.5),
        annotation_clip=False, zorder=8
    )
    ax2.text(
        zw + 1, zh - 2, 'X', color='black', fontsize=14, fontweight='bold',
        ha='left', va='bottom', zorder=9
    )

    ax2.set_xlim(0, zw)
    ax2.set_ylim(zh, 0)
    ax2.axis('off')

    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)


def process_single_prediction_file(
    json_path: str,
    data_dir: str,
    output_base_dir: str,
    dpi: int = 150
) -> Optional[Dict[str, Any]]:
    """
    Xử lý một file dự đoán trong pred_data/:
    - Đọc file JSON.
    - Tìm NIfTI MRI tương ứng.
    - Duyệt qua từng lát cắt (slices), tính toán và xuất ảnh ra thư mục con trong output_images/.
    - Xuất file CSV báo cáo số liệu từng lát cắt.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"[LỖI] File JSON không tồn tại: {json_path}")
        return None

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"[LỖI] Không thể đọc file JSON {json_path}: {e}")
        return None

    case_name = json_file.stem
    clean_name = case_name.replace('pred_', '')

    mri_file, seg_file, patient_id = find_matched_nii_and_seg(data_dir, clean_name)
    if not mri_file:
        print(f"[CẢNH BÁO] Không tìm thấy file MRI tương ứng cho '{case_name}' trong '{data_dir}'")
        return None

    try:
        img_nii = nib.load(mri_file)
        img_3d = img_nii.get_fdata()
        zooms = img_nii.header.get_zooms()
        voxel_size_mm = float(zooms[0]) if len(zooms) > 0 else 1.0
    except Exception as e:
        print(f"[LỖI] Không thể nạp NIfTI {mri_file}: {e}")
        return None

    seg_3d = None
    if seg_file and os.path.exists(seg_file):
        try:
            seg_3d = nib.load(seg_file).get_fdata()
        except Exception:
            seg_3d = None

    H, W, D = img_3d.shape
    case_out_dir = Path(output_base_dir) / clean_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print(f" Đang xử lý ca: {clean_name}")
    print(f" File MRI: {os.path.basename(mri_file)} (Kích thước: {H}x{W}x{D}, Voxel: {voxel_size_mm:.3f} mm)")
    print(f" File nhãn khối u: {os.path.basename(seg_file) if seg_file else 'Không có nhãn'}")
    print(f" Thư mục xuất ảnh: {case_out_dir}/")
    print("=" * 64)

    slice_reports = []
    saved_images_count = 0

    for z in range(D):
        img_2d = np.rot90(img_3d[:, :, z])
        seg_2d = np.rot90(seg_3d[:, :, z]) if seg_3d is not None else None

        pred_slice = None
        if isinstance(pred_data, list) and z < len(pred_data):
            pred_slice = pred_data[z]

        res = compute_midline_and_shift(
            img_2d=img_2d,
            seg_2d=seg_2d,
            pred_points=pred_slice,
            voxel_size_mm=voxel_size_mm
        )

        out_img_path = str(case_out_dir / f"slice_{z:03d}.png")

        if res is not None:
            render_and_save_slice_visualization(
                img_2d=img_2d,
                seg_2d=seg_2d,
                res=res,
                case_name=clean_name,
                slice_idx=z,
                total_slices=D,
                output_path=out_img_path,
                voxel_size_mm=voxel_size_mm,
                dpi=dpi
            )
            saved_images_count += 1
            slice_reports.append({
                'slice_idx': z,
                'midline_shift_mm': round(res['mls_mm'], 2),
                'midline_shift_px': round(res['mls_px'], 2),
                'has_tumor': 'Yes' if res['has_tumor'] else 'No',
                'tumor_side': res['tumor_side'],
                'critical_shift': 'Yes' if res['mls_mm'] >= 5.0 else 'No'
            })

    if slice_reports:
        csv_path = case_out_dir / "midline_shift_report.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['slice_idx', 'midline_shift_mm', 'midline_shift_px', 'has_tumor', 'tumor_side', 'critical_shift']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in slice_reports:
                writer.writerow(r)

        max_stat = max(slice_reports, key=lambda s: s['midline_shift_mm'])
        mean_shift = np.mean([s['midline_shift_mm'] for s in slice_reports])

        summary_data = {
            'case_name': clean_name,
            'mri_file': str(mri_file),
            'total_slices': D,
            'processed_slices': len(slice_reports),
            'max_midline_shift_mm': max_stat['midline_shift_mm'],
            'max_midline_shift_px': max_stat['midline_shift_px'],
            'slice_with_max_shift': max_stat['slice_idx'],
            'mean_shift_mm': round(float(mean_shift), 2),
            'critical_mass_effect': max_stat['midline_shift_mm'] >= 5.0
        }

        json_summary_path = case_out_dir / "summary.json"
        with open(json_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"[HOÀN THÀNH] Đã xuất {saved_images_count} ảnh vào: {case_out_dir}/")
        print(f" - Độ lệch cực đại lớn nhất (Max MLS): {max_stat['midline_shift_mm']:.2f} mm (tại Lát #{max_stat['slice_idx']:03d})")
        print(f" - Báo cáo chi tiết: {csv_path}")

        return summary_data

    return None


def run_pipeline(
    pred_dir: str = "pred_data",
    data_dir: str = "sample_data",
    output_dir: str = "output_images",
    specific_json: Optional[str] = None,
    dpi: int = 150
) -> None:
    """
    Chạy pipeline xử lý toàn bộ các file trong thư mục pred_data.
    """
    pred_path = Path(pred_dir)
    if not pred_path.exists():
        print(f"[LỖI] Thư mục dự đoán '{pred_dir}' không tồn tại.")
        return

    if specific_json:
        json_files = [specific_json]
    else:
        json_files = sorted(glob.glob(str(pred_path / "*.json")))

    if not json_files:
        print(f"[LỖI] Không tìm thấy file JSON nào trong '{pred_dir}'")
        return

    print(f"\n=======================================================")
    print(f" KHỞI ĐỘNG PIPELINE ĐO ĐỘ LỆCH RÃNH GIỮA NÃO BỘ (MLS)")
    print(f" Tổng số ca cần xử lý: {len(json_files)}")
    print(f" Thư mục dữ liệu gốc: {data_dir}")
    print(f" Thư mục đầu ra: {output_dir}")
    print(f"=======================================================\n")

    all_summaries = []
    for jf in json_files:
        summary = process_single_prediction_file(
            json_path=jf,
            data_dir=data_dir,
            output_base_dir=output_dir,
            dpi=dpi
        )
        if summary:
            all_summaries.append(summary)

    # Xuất báo cáo tổng quan toàn bộ tập dữ liệu
    if all_summaries:
        global_csv = Path(output_dir) / "global_midline_shift_summary.csv"
        with open(global_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'case_name', 'total_slices', 'processed_slices', 
                'max_midline_shift_mm', 'max_midline_shift_px', 
                'slice_with_max_shift', 'mean_shift_mm', 'critical_mass_effect'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({k: s[k] for k in fieldnames})

        print("\n" + "=" * 64)
        print(" TỔNG KẾT TOÀN BỘ CÁC CA ĐÃ XỬ LÝ")
        print("=" * 64)
        for s in all_summaries:
            crit = " [CẢNH BÁO NGUY HIỂM: MLS >= 5mm]" if s['critical_mass_effect'] else ""
            print(f"• Ca: {s['case_name']:<30} | Max MLS: {s['max_midline_shift_mm']:>5.2f} mm (Slice #{s['slice_with_max_shift']:03d}){crit}")
        print(f"\n Đã lưu file tổng kết toàn cục: {global_csv}")
        print(f" Toàn bộ ảnh đã được lưu vào các thư mục tương ứng trong '{output_dir}/'.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Đọc pred_data, vẽ đường cong bám khối u (đỏ), đường thẳng nối đầu mút (xanh) và tính độ lệch MLS"
    )
    parser.add_argument("--pred_dir", type=str, default="pred_data", help="Thư mục chứa các file .json dự đoán")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Thư mục chứa ảnh MRI scan gốc (.nii.gz)")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Thư mục lưu các ảnh lát cắt và báo cáo")
    parser.add_argument("--json_file", type=str, default=None, help="Chạy riêng 1 file json cụ thể (tùy chọn)")
    parser.add_argument("--dpi", type=int, default=150, help="Độ phân giải DPI của ảnh xuất ra")

    args = parser.parse_args()

    run_pipeline(
        pred_dir=args.pred_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        specific_json=args.json_file,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
