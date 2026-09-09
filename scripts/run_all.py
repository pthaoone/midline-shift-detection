#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Runner for Midline Shift Detection
======================================================
Thực hiện toàn bộ chu trình tự động:
1. Tạo nhãn train_data từ sample_data (nếu chưa có).
2. Huấn luyện model_msd.pt bằng scripts/train.py.
3. Chạy dự đoán toàn bộ ca bệnh vào pred_data/.
4. Chạy scripts/visualize.py xuất ảnh vào output_images/<case_name>/.
5. Đánh giá sai số & Accuracy bằng scripts/evaluate_metrics.py.
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path

# Configure utf-8 encoding for Windows terminal
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_command(cmd_list, desc):
    print(f"\n>>> [BƯỚC] {desc}...")
    print(f"Lệnh thực thi: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, text=True)
    if res.returncode != 0:
        print(f"[!] Cảnh báo: Lệnh kết thúc với mã lỗi {res.returncode}")
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Chạy tự động toàn bộ pipeline Midline Shift Detection")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Thư mục chứa các ca bệnh gốc")
    parser.add_argument("--train_dir", type=str, default="train_data", help="Thư mục dữ liệu huấn luyện")
    parser.add_argument("--pred_dir", type=str, default="pred_data", help="Thư mục xuất kết quả dự đoán")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Thư mục xuất ảnh trực quan hóa")
    parser.add_argument("--model", type=str, default="model_msd.pt", help="File lưu trọng số mô hình")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batches", type=int, default=10, help="Số batch mỗi epoch")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị tính toán (cpu hoặc cuda)")
    parser.add_argument("--max_predict_cases", type=int, default=10, help="Số lượng ca dự đoán thử nghiệm (mặc định 10 ca)")
    parser.add_argument("--case", type=str, default=None, help="Chỉ chạy dự đoán và trực quan hóa cho 1 ca cụ thể (ví dụ: BraTS-MEN-00023-000)")
    parser.add_argument("--skip_train", action="store_true", help="Bỏ qua bước huấn luyện nếu đã có sẵn file model")
    args = parser.parse_args()

    python_exec = sys.executable

    # BƯỚC 1: Chuẩn bị dữ liệu nhãn train_data
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    existing_pairs = list(Path(args.train_dir).glob("*.json"))
    if not existing_pairs:
        run_command([
            python_exec, "scripts/prepare_dataset.py",
            "--data_dir", args.data_dir,
            "--out_dir", args.train_dir,
            "--modality", "t1c"
        ], "1. Tạo file nhãn .json cho tập huấn luyện")
    else:
        print(f"\n>>> [BƯỚC 1] Đã có sẵn {len(existing_pairs)} ca trong '{args.train_dir}', bỏ qua bước tạo nhãn.")

    # BƯỚC 2: Huấn luyện mô hình
    if args.skip_train and os.path.exists(args.model):
        print(f"\n>>> [BƯỚC 2] Bỏ qua huấn luyện, sử dụng mô hình sẵn có: '{args.model}'")
    else:
        run_command([
            python_exec, "scripts/train.py",
            args.train_dir, args.model,
            "--epochs", str(args.epochs),
            "--batches_per_epoch", str(args.batches),
            "--batch_size", "2",
            "--device", args.device
        ], "2. Huấn luyện mô hình CNN (scripts/train.py)")

    # BƯỚC 3: Dự đoán các ca và lưu vào pred_data/
    Path(args.pred_dir).mkdir(parents=True, exist_ok=True)
    if args.case:
        case_dirs = [d for d in Path(args.data_dir).iterdir() if d.is_dir() and args.case in d.name]
    else:
        case_dirs = [d for d in Path(args.data_dir).iterdir() if d.is_dir()]
        if args.max_predict_cases:
            case_dirs = case_dirs[:args.max_predict_cases]

    print(f"\n>>> [BƯỚC 3] Dự đoán bằng mô hình '{args.model}' cho {len(case_dirs)} ca vào '{args.pred_dir}/'...")
    for c in case_dirs:
        mri_files = list(c.glob("*-t1c.nii.gz")) or [f for f in c.glob("*.nii.gz") if not f.name.endswith("-seg.nii.gz")]
        if mri_files:
            in_mri = str(mri_files[0])
            out_json = str(Path(args.pred_dir) / f"{c.name}.json")
            subprocess.run([
                python_exec, "scripts/predict.py",
                in_mri, out_json, args.model,
                "--device", args.device
            ], capture_output=True)
            print(f" [+] Đã dự đoán thành công: {c.name} -> {out_json}")

    # BƯỚC 4: Trực quan hóa ảnh não, vẽ đường đỏ bám u, đường xanh nối cực và đo đạc độ lệch MLS
    vis_cmd = [
        python_exec, "scripts/visualize_midline_shift.py",
        "--pred_dir", args.pred_dir,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ]
    if args.case:
        specific_json = Path(args.pred_dir) / f"{args.case}.json"
        if specific_json.exists():
            vis_cmd.extend(["--json_file", str(specific_json)])

    run_command(vis_cmd, "4. Trực quan hóa ảnh não & Đo độ lệch MLS (scripts/visualize_midline_shift.py)")

    # BƯỚC 5: Đánh giá đầy đủ 10 thông số chuẩn số thập phân
    run_command([
        python_exec, "scripts/evaluate_metrics.py",
        "--gt_dir", args.train_dir,
        "--pred_dir", args.pred_dir,
        "--data_dir", args.data_dir
    ], "5. Tính toán và hiển thị đầy đủ 10 thông số dạng số thập phân (scripts/evaluate_metrics.py)")

    print("\n" + "=" * 60)
    print(" [HOÀN TẤT TOÀN BỘ PIPELINE]")
    print(f" • Mô hình đã lưu: {args.model}")
    print(f" • Tọa độ dự đoán: {args.pred_dir}/")
    print(f" • Ảnh trực quan hóa: {args.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
