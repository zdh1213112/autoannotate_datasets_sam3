"""Detection selection and YOLO export for the single-barcode workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUTO_SCORE = 0.80
COMPETING_SCORE = 0.45


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def normalize_roi(roi):
    try:
        valid_length = roi is not None and len(roi) == 4
    except TypeError:
        valid_length = False
    if not valid_length:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in roi)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2] if x2 - x1 >= 0.01 and y2 - y1 >= 0.01 else None


def roi_pixels(roi, width: int, height: int) -> tuple[int, int, int, int]:
    roi = normalize_roi(roi)
    if roi is None:
        raise ValueError("请先框选有效的标注区域")
    x1 = max(0, min(width - 1, int(np.floor(roi[0] * width))))
    y1 = max(0, min(height - 1, int(np.floor(roi[1] * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil(roi[2] * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(roi[3] * height))))
    return x1, y1, x2, y2


def _candidate_polygon(mask, box, crop_shape):
    """Fit a rotated box to the segmented barcode, with its box as fallback."""
    height, width = crop_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    points = None
    if mask is not None:
        binary = np.asarray(mask, dtype=np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        restricted = np.zeros_like(binary)
        xa, ya = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        xb, yb = int(min(width, np.ceil(x2))), int(min(height, np.ceil(y2)))
        restricted[ya:yb, xa:xb] = binary[ya:yb, xa:xb]
        points = cv2.findNonZero(restricted)

    if points is not None and len(points) >= 20:
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        # A little margin covers light quiet zones and weak mask edges.
        rect = (center, (size[0] * 1.06 + 2, size[1] * 1.06 + 2), angle)
        polygon = cv2.boxPoints(rect)
    else:
        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon if cv2.contourArea(polygon) >= 100 else None


def select_barcode(image, roi, masks, boxes, scores) -> dict:
    """Choose at most one barcode from SAM3 results on the cropped ROI."""
    height, width = image.shape[:2]
    rx1, ry1, rx2, ry2 = roi_pixels(roi, width, height)
    crop_shape = (ry2 - ry1, rx2 - rx1)
    candidates = []
    for index, box in enumerate(boxes):
        if len(box) != 4:
            continue
        mask = masks[index] if index < len(masks) else None
        polygon = _candidate_polygon(mask, box, crop_shape)
        if polygon is None:
            continue
        polygon += np.array([rx1, ry1], dtype=np.float32)
        score = float(scores[index]) if index < len(scores) else 0.0
        if not np.isfinite(score):
            continue
        candidates.append((score, polygon))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return {"polygon": None, "score": 0.0, "status": "review",
                "reason": "区域内未找到条形码，请人工补框或确认无目标", "candidates": 0}

    score, polygon = candidates[0]
    rivals = [other for other_score, other in candidates[1:]
              if other_score >= COMPETING_SCORE and
              cv2.intersectConvexConvex(polygon, other)[0] /
              max(1.0, min(cv2.contourArea(polygon), cv2.contourArea(other))) < 0.5]
    near_edge = bool(np.any(polygon[:, 0] <= rx1 + 2) or
                     np.any(polygon[:, 0] >= rx2 - 3) or
                     np.any(polygon[:, 1] <= ry1 + 2) or
                     np.any(polygon[:, 1] >= ry2 - 3))
    if rivals:
        status, reason = "review", "区域内有多个独立候选，请人工选择唯一条形码"
    elif near_edge:
        status, reason = "review", "条形码接近标注区域边缘，请人工检查"
    elif score < AUTO_SCORE:
        status, reason = "review", "模型置信度较低，请人工检查"
    else:
        status, reason = "auto", "高置信度自动候选，可人工复核"
    return {"polygon": polygon.round(2).tolist(), "score": round(score, 4),
            "status": status, "reason": reason, "candidates": len(candidates)}


def detect_barcode(image, roi, segmenter) -> dict:
    x1, y1, x2, y2 = roi_pixels(roi, image.shape[1], image.shape[0])
    masks, boxes, scores = segmenter.detect(image[y1:y2, x1:x2], "barcode")
    return select_barcode(image, roi, masks, boxes, scores)


def polygon_in_roi(polygon, roi, width, height) -> bool:
    if polygon is None:
        return False
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if cv2.contourArea(points) < 25:
        return False
    x1, y1, x2, y2 = roi_pixels(roi, width, height)
    cx, cy = points.mean(axis=0)
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):
        return False
    roi_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       dtype=np.float32)
    overlap, _ = cv2.intersectConvexConvex(points, roi_box)
    return overlap / cv2.contourArea(points) >= 0.5


def yolo_label(polygon, width: int, height: int, box_format: str) -> str:
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("条形码框无效")
    if box_format == "obb":
        normalized = points / np.array([width, height], dtype=np.float32)
        numbers = normalized.reshape(-1)
    elif box_format == "hbb":
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        numbers = np.array([(lower[0] + upper[0]) / (2 * width),
                            (lower[1] + upper[1]) / (2 * height),
                            (upper[0] - lower[0]) / width,
                            (upper[1] - lower[1]) / height])
    else:
        raise ValueError(f"未知标签格式: {box_format}")
    if np.any(numbers < 0) or np.any(numbers > 1):
        raise ValueError("条形码框超出图片边界")
    return "0 " + " ".join(f"{float(value):.6f}" for value in numbers) + "\n"


def load_records(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record["name"]] = record
    return records


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_dataset(source_dir: Path, output_dir: Path, records: dict[str, dict],
                   box_format: str) -> tuple[Path, int, int]:
    """Create a new, self-contained export; never replace an earlier export."""
    selected = [record for record in records.values()
                if record["status"] in {"auto", "confirmed", "empty_confirmed"}]
    if len(selected) < 2:
        raise ValueError("至少需要 2 张可导出的图片，才能建立训练集和验证集")
    selected.sort(key=lambda item: (hashlib.sha256(item["name"].encode()).hexdigest(),
                                    item["name"]))
    val_count = min(len(selected) - 1, max(1, round(len(selected) * 0.2)))
    val_names = {record["name"] for record in selected[:val_count]}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_dir / f"yolo_{box_format}_{stamp}"
    suffix = 2
    while target.exists():
        target = output_dir / f"yolo_{box_format}_{stamp}_{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    seen_stems = set()
    positive = 0
    for record in sorted(selected, key=lambda item: item["name"]):
        name = record["name"]
        source = source_dir / name
        if not source.is_file() or Path(name).name != name:
            raise FileNotFoundError(f"源图片不存在: {source}")
        stem = Path(name).stem
        if stem in seen_stems:
            raise ValueError(f"不同扩展名的图片共用文件名: {stem}")
        seen_stems.add(stem)
        subset = "val" if name in val_names else "train"
        image_dir = target / "images" / subset
        label_dir = target / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        polygon = record.get("polygon")
        if polygon is None:
            label = ""
        else:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"无法读取图片: {source}")
            label = yolo_label(polygon, image.shape[1], image.shape[0], box_format)
            positive += 1
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    (target / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: barcode\n",
        encoding="utf-8",
    )
    (target / "README.txt").write_text(
        "Single-barcode dataset. Empty label files are manually confirmed negatives.\n"
        "Source images were hard-linked when possible and copied otherwise.\n",
        encoding="utf-8",
    )
    return target, positive, len(selected) - positive
