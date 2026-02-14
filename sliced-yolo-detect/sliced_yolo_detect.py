# Author: Naoki
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from typing import Optional, Tuple
from collections import defaultdict

import argparse
import time

# CUDA利用可否を判定（使えなければCPUにフォールバック）
try:
    import torch

    _cuda_ok = bool(torch.cuda.is_available())
except Exception:
    _cuda_ok = False

device = "cuda:0" if _cuda_ok else "cpu"

_start_time = time.perf_counter()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="path to model file")
    p.add_argument("--image-path", required=True, help="path to input image")
    p.add_argument("--slice-height", type=int, default=512)
    p.add_argument("--slice-width", type=int, default=512)
    p.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.2,
        help="slice overlap ratio for both height/width (default: 0.2)",
    )
    p.add_argument("--confidence", type=float, default=0.6)
    p.add_argument("--yolo-conf", type=float, default=0.2, help="YOLO confidence_threshold (default: 0.2)")
    p.add_argument("--show-size", action="store_true", help="draw bbox W/H (px) and print per-class bbox size stats")
    p.add_argument("--w-range", default=None, help="bbox width range in pixels: w1-w2 (inclusive)")
    p.add_argument("--h-range", default=None, help="bbox height range in pixels: h1-h2 (inclusive)")
    p.add_argument("--csv", default=None, help="write detections to CSV (cls,bbox_w,bbox_h,conf)")
    return p.parse_args()

args = parse_args()
model_path = args.model_path
image_path = args.image_path

# Validate input paths
import os
import sys

if not os.path.isfile(model_path):
    print(f"Error: model file not found: {model_path}", file=sys.stderr)
    raise SystemExit(2)

if not os.path.isfile(image_path):
    print(f"Error: image file not found: {image_path}", file=sys.stderr)
    raise SystemExit(2)

# 推論設定（YOLO側のconfidence_threshold）
YOLO_CONFIDENCE_THRESHOLD = float(args.yolo_conf)

if args.confidence < YOLO_CONFIDENCE_THRESHOLD:
    print(
        f"Warning: --confidence ({args.confidence}) must be higher than --yolo-conf ({YOLO_CONFIDENCE_THRESHOLD}).",
        file=__import__("sys").stderr,
    )
    raise SystemExit(2)

# 可視化・情報表示のため、先に画像を読み込んでサイズを取得（OpenCVではなくPillow）
import hashlib
from PIL import Image, ImageDraw, ImageFont

# Pillowで巨大画像を扱う（必要なら制限解除）
Image.MAX_IMAGE_PIXELS = None

im = Image.open(image_path).convert("RGB")
w, h = im.size

param_lines = [
    "--- params ---",
    f"GPU_available: {_cuda_ok}",
    f"device: {device}",
    f"model: {os.path.basename(model_path)}",
    f"image: {os.path.basename(image_path)}",
    f"image_size: {w} x {h}",
    f"slice_height: {args.slice_height}",
    f"slice_width: {args.slice_width}",
    f"overlap_ratio: {args.overlap_ratio}",
    f"confidence limit (for counting): {args.confidence}",
    f"yolo_confidence_threshold: {YOLO_CONFIDENCE_THRESHOLD}",
    f"width range limit (for counting): {args.w_range}",
    f"height range limit (for counting): {args.h_range}",
    "-------------",
]
for line in param_lines:
    print(line)

# モデル作成

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    device=device,
)

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=args.slice_height,
    slice_width=args.slice_width,
    overlap_height_ratio=args.overlap_ratio,
    overlap_width_ratio=args.overlap_ratio,
)

print("SAHI processing completed. Preparing visualization...")
_elapsed_so_far = time.perf_counter() - _start_time
print(f"elapsed so far: {_elapsed_so_far:.3f} sec")

# 可視化（1枚書き出し）
# OpenCVではなくPillowで1px描画する

# class名 -> RGB色 を安定して割り当て
# 先頭4クラスは固定パレット: 1=青, 2=明るいシアン, 3=黄, 4=明るい緑
_CLASS_COLOR_PALETTE_RGB = [
    (0, 0, 255),      # blue
    (0, 255, 255),    # bright cyan
    (255, 255, 0),    # yellow
    (0, 255, 0),      # bright green
]
_class_color_map = {}


def _fallback_color_from_name(name: str) -> Tuple[int, int, int]:
    # ハッシュから色相を作り、HSV->RGBで色の差を出す
    import colorsys

    d = hashlib.md5(name.encode("utf-8")).digest()
    hue = d[0] / 255.0  # 0..1
    sat = 0.85
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def class_to_rgb(cls_name: Optional[str]) -> Tuple[int, int, int]:
    if not cls_name:
        return (0, 255, 0)

    if cls_name in _class_color_map:
        return _class_color_map[cls_name]

    idx = len(_class_color_map)
    if idx < len(_CLASS_COLOR_PALETTE_RGB):
        color = _CLASS_COLOR_PALETTE_RGB[idx]
    else:
        color = _fallback_color_from_name(cls_name)

    _class_color_map[cls_name] = color
    return color


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return (x1 - x0), (y1 - y0)
    except Exception:
        return draw.textsize(text, font=font)


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


draw = ImageDraw.Draw(im)
font_label = _load_font(11)  # class/conf
font_size = _load_font(11)   # W/H
LINE_WIDTH = 1

os.makedirs("out", exist_ok=True)

_size_stats = defaultdict(lambda: {"w": [], "h": []})


def _histogram_10px(values_px):
    if not values_px:
        return []
    max_v = int(max(values_px))
    end = ((max_v // 10) + 1) * 10
    bins = [0] * (end // 10)
    for v in values_px:
        iv = int(v)
        idx = iv // 10
        if idx < 0:
            idx = 0
        if idx >= len(bins):
            idx = len(bins) - 1
        bins[idx] += 1
    out = []
    for i, c in enumerate(bins):
        lo = i * 10
        hi = lo + 10
        out.append((f"{lo:>4}-{hi:<4}", c))
    return out


def _format_histogram_bars(hist, max_bar_width: int = 40):
    if not hist:
        return []
    max_count = max(c for _, c in hist) or 1
    lines = []
    for label, count in hist:
        if count == 0:
            continue
        bar_len = int(round(count * max_bar_width / max_count)) if count > 0 else 0
        lines.append((label, count, "*" * bar_len))
    return lines


def _parse_range(spec: Optional[str], name: str) -> Optional[Tuple[int, int]]:
    if not spec:
        return None
    try:
        a, b = spec.split("-", 1)
        lo = int(a)
        hi = int(b)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    except Exception:
        print(f"Warning: invalid {name} '{spec}'. expected like '10-200'", file=__import__("sys").stderr)
        raise SystemExit(2)


w_range = _parse_range(args.w_range, "--w-range")
h_range = _parse_range(args.h_range, "--h-range")

for op in result.object_prediction_list:
    x1, y1, x2, y2 = map(int, op.bbox.to_xyxy())

    # 画像境界に丸める
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)

    # class 名
    cls = None
    try:
        cls = op.category.name
    except Exception:
        try:
            cls = str(op.category.id)
        except Exception:
            cls = None

    # confidence
    conf = None
    try:
        conf = float(op.score.value)
    except Exception:
        try:
            conf = float(op.score)
        except Exception:
            conf = None

    # 色: confが--confidence未満 もしくは（指定がある場合）bboxサイズが範囲外なら赤
    out_of_range = False
    if w_range is not None:
        w1, w2 = w_range
        if bbox_w < w1 or bbox_w > w2:
            out_of_range = True
    if h_range is not None:
        h1, h2 = h_range
        if bbox_h < h1 or bbox_h > h2:
            out_of_range = True

    if (conf is not None and conf < args.confidence) or out_of_range:
        color = (255, 0, 0)  # red (RGB)
    else:
        color = class_to_rgb(cls)

    # rectangle outline
    draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

    if args.show_size:
        key = cls if cls else "unknown"
        _size_stats[key]["w"].append(bbox_w)
        _size_stats[key]["h"].append(bbox_h)

        # 下辺中央に幅(px)
        w_text = str(bbox_w)
        tw, th = _text_size(draw, w_text, font_size)
        x_center = x1 + max(0, (bbox_w - tw) // 2)
        y_bottom = min(h - th, y2 + 2)
        x_center = _clamp(x_center, 0, max(0, w - tw))
        y_bottom = _clamp(y_bottom, 0, max(0, h - th))
        draw.text((x_center, y_bottom), w_text, fill=color, font=font_size)

        # 右辺中央に高さ(px)
        h_text = str(bbox_h)
        tw2, th2 = _text_size(draw, h_text, font_size)
        x_right = min(w - tw2, x2 + 2)
        y_center = y1 + max(0, (bbox_h - th2) // 2)
        x_right = _clamp(x_right, 0, max(0, w - tw2))
        y_center = _clamp(y_center, 0, max(0, h - th2))
        draw.text((x_right, y_center), h_text, fill=color, font=font_size)

    # class + conf label
    if cls is not None or conf is not None:
        if cls is None:
            text = f"{conf:.2f}" if conf is not None else ""
        elif conf is None:
            text = f"{cls}"
        else:
            text = f"{cls} {conf:.2f}"

        if text:
            ty = y1 - 12
            if ty < 0:
                ty = 0
            draw.text((x1, ty), text, fill=color, font=font_label)

# --- 集計（conf>=YOLO_CONFIDENCE_THRESHOLD と conf>=--confidence をクラス毎にカウント）
counts_02 = {}
counts_user = {}

for op in result.object_prediction_list:
    cls = None
    try:
        cls = op.category.name
    except Exception:
        try:
            cls = str(op.category.id)
        except Exception:
            cls = None
    if not cls:
        cls = "unknown"

    conf = None
    try:
        conf = float(op.score.value)
    except Exception:
        try:
            conf = float(op.score)
        except Exception:
            conf = None

    if conf is None:
        continue

    if conf >= YOLO_CONFIDENCE_THRESHOLD:
        counts_02[cls] = counts_02.get(cls, 0) + 1

    # counts_user: conf>=--confidence に加えて、指定がある場合は bbox W/H 範囲でもフィルタ
    if conf >= args.confidence:
        x1, y1, x2, y2 = map(int, op.bbox.to_xyxy())
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)

        if w_range is not None:
            w1, w2 = w_range
            if not (w1 <= bbox_w <= w2):
                continue
        if h_range is not None:
            h1, h2 = h_range
            if not (h1 <= bbox_h <= h2):
                continue

        counts_user[cls] = counts_user.get(cls, 0) + 1

print(f"detections (conf >= {YOLO_CONFIDENCE_THRESHOLD:.3f}) by class:")
for cls_name in sorted(counts_02.keys()):
    print(f"  {cls_name}: {counts_02[cls_name]}")
print("total:", sum(counts_02.values()))

# ユーザ指定の幅/高さ範囲がある場合はその旨を付記して出力
hdr = f"detections (conf >= {args.confidence:.3f}) by class:"
filters = []
if w_range is not None:
    filters.append(f"width={w_range[0]}-{w_range[1]}px")
if h_range is not None:
    filters.append(f"height={h_range[0]}-{h_range[1]}px")
if filters:
    hdr += " (filter: " + ", ".join(filters) + ")"
print(hdr)
for cls_name in sorted(counts_user.keys()):
    print(f"  {cls_name}: {counts_user[cls_name]}")
print("total:", sum(counts_user.values()))

if args.show_size:
    print("\n--- bbox size stats (px) ---")
    for cls_name in sorted(_size_stats.keys()):
        ws = _size_stats[cls_name]["w"]
        hs = _size_stats[cls_name]["h"]
        if not ws or not hs:
            continue

        w_min, w_max = min(ws), max(ws)
        h_min, h_max = min(hs), max(hs)
        w_avg = sum(ws) / len(ws)
        h_avg = sum(hs) / len(hs)

        print(f"Class {cls_name}:")
        print(f"  width : min={w_min} max={w_max} avg={w_avg:.1f}")
        print(f"  height: min={h_min} max={h_max} avg={h_avg:.1f}")

        print("  width histogram (10px bins):")
        for label, count, bar in _format_histogram_bars(_histogram_10px(ws)):
            if count == 0:
                continue
            print(f"    {label}: {count:>6} {bar}")

        print("  height histogram (10px bins):")
        for label, count, bar in _format_histogram_bars(_histogram_10px(hs)):
            if count == 0:
                continue
            print(f"    {label}: {count:>6} {bar}")
        print()

if args.show_size:
    # CLI出力と同じ params を左上にオーバーレイ
    overlay_margin = 8
    overlay_pad = 4
    line_gap = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    line_sizes = [_text_size(draw, line, font_label) for line in param_lines]
    max_w = max((tw for tw, _ in line_sizes), default=0)
    total_h = sum(th for _, th in line_sizes) + max(0, len(line_sizes) - 1) * line_gap

    left = overlay_margin
    top = overlay_margin
    right = min(w - 1, left + max_w + overlay_pad * 2)
    bottom = min(h - 1, top + total_h + overlay_pad * 2)
    draw.rectangle([left, top, right, bottom], fill=bg_color)

    tx = left + overlay_pad
    ty = top + overlay_pad
    for line, (_, th) in zip(param_lines, line_sizes):
        draw.text((tx, ty), line, fill=text_color, font=font_label)
        ty += th + line_gap

# CSV出力（指定された場合）
if args.csv:
    import csv

    csv_path = args.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["class", "class_id", "confidence", "x1", "x2", "y1", "y2", "bbox_w", "bbox_h"])
        for op in result.object_prediction_list:
            # class name / class id
            cls_name = None
            cls_id = None
            try:
                cls_name = op.category.name
            except Exception:
                cls_name = None
            try:
                cls_id = op.category.id
            except Exception:
                cls_id = None
            if not cls_name:
                try:
                    cls_name = str(op.category.id)
                except Exception:
                    cls_name = "unknown"
            if cls_id is None and cls_name != "unknown":
                try:
                    cls_id = int(cls_name)
                except Exception:
                    cls_id = ""
            elif cls_id is None:
                cls_id = ""

            # bbox coords / size
            x1, y1, x2, y2 = map(int, op.bbox.to_xyxy())
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)

            # confidence
            conf = None
            try:
                conf = float(op.score.value)
            except Exception:
                try:
                    conf = float(op.score)
                except Exception:
                    conf = None

            wtr.writerow(
                [
                    cls_name,
                    cls_id,
                    "" if conf is None else f"{conf:.6f}",
                    x1,
                    x2,
                    y1,
                    y2,
                    bbox_w,
                    bbox_h,
                ]
            )
    print("csv saved:", csv_path)

# 保存ファイル名: inputのファイル名 + "_detect" (+ 連番) + 拡張子
base_name = os.path.basename(image_path)
stem, ext = os.path.splitext(base_name)
if not ext:
    ext = ".png"

out_dir = "out"
os.makedirs(out_dir, exist_ok=True)

candidate = os.path.join(out_dir, f"{stem}_detect{ext}")
idx = 1
while os.path.exists(candidate):
    candidate = os.path.join(out_dir, f"{stem}_detect{idx}{ext}")
    idx += 1

out_path = candidate

# 保存（Pillow）
im.save(out_path)
print("saved:", out_path)

_elapsed = time.perf_counter() - _start_time
print(f"elapsed: {_elapsed:.3f} sec")
