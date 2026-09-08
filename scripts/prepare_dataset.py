#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Dataset & Generate Annotations for Midline Shift Detection
===================================================================
Script này tự động:
1. Quét các ca bệnh trong `sample_data/`.
2. Trích xuất rãnh giữa não bộ và hiệu ứng chèn ép khối u từ `-seg.nii.gz`
   để sinh các file nhãn `.json` chuẩn theo format của bài báo MSD.
3. Xuất tập dữ liệu huấn luyện sẵn sàng cho `scripts/train.py`.
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

# Configure utf-8 encoding for Windows terminal
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes, gaussian_filter1d


def extract_midline_curve_from_slice(img_2d: np.ndarray, seg_2d: np.ndarray = None) -> list:
    """
    Trích xuất danh sách các điểm tọa độ (y, x) của đường giữa cho 1 lát cắt 2D.
    Nếu lát cắt không chứa mô não, trả về [] (empty list).
    """
    H, W = img_2d.shape
    nonzero = img_2d[img_2d > 0]
    if len(nonzero) < 150:
        return []

    thresh = np.percentile(nonzero, 8) if len(nonzero) > 0 else 0
    brain_mask = binary_fill_holes(img_2d > thresh)
    y_coords, x_coords = np.where(brain_mask)
    if len(y_coords) < 150 or (np.max(y_coords) - np.min(y_coords)) < 20:
        return []

    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    margin_y = max(3, int((y_max - y_min) * 0.035))
    y_ant, y_post = y_min + margin_y, y_max - margin_y

    ant_pts = np.where(brain_mask[y_ant, :])[0]
    post_pts = np.where(brain_mask[y_post, :])[0]
    if len(ant_pts) == 0 or len(post_pts) == 0:
        return []

    x_ant = float(np.mean(ant_pts))
    x_post = float(np.mean(post_pts))

    # Xây dựng đường thẳng lý tưởng AML
    y_range = np.arange(y_ant, y_post + 1)
    t = (y_range - y_ant) / max(1e-6, (y_post - y_ant))
    ideal_x = (1.0 - t) * x_ant + t * x_post
    def_x = ideal_x.copy()

    # Nếu có khối u ở lát cắt này, tính độ cong do u chèn ép
    if seg_2d is not None and np.any(seg_2d > 0):
        t_y, t_x = np.where(seg_2d > 0)
        t_cy = float(np.mean(t_y))
        t_cx = float(np.mean(t_x))
        ideal_cx = float(np.interp(t_cy, y_range, ideal_x))
        tumor_side = "Right" if t_cx > ideal_cx else "Left"

        for i, y_val in enumerate(y_range):
            row_tumor = np.where(seg_2d[y_val, :] > 0)[0]
            if len(row_tumor) > 0:
                if tumor_side == "Right":
                    medial_x = float(np.min(row_tumor))
                    dist = max(0.0, medial_x - ideal_x[i])
                    shift = max(2.0, 14.0 * np.exp(-dist / 25.0))
                    def_x[i] = ideal_x[i] - shift
                else:
                    medial_x = float(np.max(row_tumor))
                    dist = max(0.0, ideal_x[i] - medial_x)
                    shift = max(2.0, 14.0 * np.exp(-dist / 25.0))
                    def_x[i] = ideal_x[i] + shift

        def_x = gaussian_filter1d(def_x, sigma=2.0)

    # Chuyển thành danh sách các điểm [[y, x], ...]
    points = [[float(y), float(x)] for y, x in zip(y_range, def_x)]
    return points


def generate_annotations_for_case(case_dir: Path, output_json_path: Path, modality: str = "t1c"):
    """
    Sinh file annotation JSON cho 1 ca chụp.
    Format JSON: [ [ curve_slice_0, curve_slice_1, ..., curve_slice_D-1 ] ]
    """
    mri_file = list(case_dir.glob(f"*-{modality}.nii.gz"))
    if not mri_file:
        mri_file = [f for f in case_dir.glob("*.nii.gz") if not f.name.endswith("-seg.nii.gz")]
    if not mri_file:
        return False

    mri_path = mri_file[0]
    seg_files = list(case_dir.glob("*-seg.nii.gz"))
    seg_path = seg_files[0] if seg_files else None

    img_nii = nib.load(str(mri_path))
    img_data = img_nii.get_fdata()
    seg_data = nib.load(str(seg_path)).get_fdata() if seg_path else None

    num_slices = img_data.shape[-1]
    single_annotation = []

    for s in range(num_slices):
        img_2d = img_data[..., s]
        seg_2d = seg_data[..., s] if seg_data is not None else None
        curve = extract_midline_curve_from_slice(img_2d, seg_2d)
        single_annotation.append(curve)

    full_annotations = [single_annotation]

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_annotations, f)

    return True


def main():
    parser = argparse.ArgumentParser(description="Tạo các file nhãn .json từ thư mục sample_data")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Thư mục chứa các ca bệnh")
    parser.add_argument("--out_dir", type=str, default="train_data", help="Thư mục xuất dữ liệu train (ảnh + json)")
    parser.add_argument("--modality", type=str, default="t1c", help="Loại ảnh MRI dùng để train (t1c, t2f, t2w, t1n)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Bắt đầu tạo nhãn .json cho {len(case_dirs)} ca trong '{args.data_dir}'...")

    success_count = 0
    for c_dir in case_dirs:
        case_name = c_dir.name
        mri_files = list(c_dir.glob(f"*-{args.modality}.nii.gz"))
        if not mri_files:
            mri_files = [f for f in c_dir.glob("*.nii.gz") if not f.name.endswith("-seg.nii.gz")]

        if mri_files:
            src_mri = mri_files[0]
            dst_mri = out_dir / f"{case_name}.nii.gz"
            dst_json = out_dir / f"{case_name}.json"

            shutil.copyfile(src_mri, dst_mri)
            if generate_annotations_for_case(c_dir, dst_json, modality=args.modality):
                success_count += 1
                print(f" [+] Đã tạo: {dst_mri.name} & {dst_json.name}")

    print(f"\n[HOÀN TẤT] Đã tạo thành công {success_count} cặp dữ liệu (ảnh .nii.gz + nhãn .json) trong '{args.out_dir}/'")


if __name__ == "__main__":
    main()
