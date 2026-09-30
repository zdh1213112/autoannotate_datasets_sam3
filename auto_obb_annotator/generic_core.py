"""Multi-class, variable-count SAM3 annotation selection and YOLO export."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from .barcode_core import polygon_in_roi, roi_pixels


AUTO_SCORE = 0.70
EXTRA_SCORE = 0.45
NMS_IOU = 0.70


def validate_classes(classes: list[dict]) -> list[dict]:
    if not classes:
        raise ValueError("请至少添加一个类别")
    result = []
    names = set()
    prompts = set()
    for spec in classes:
        name = str(spec.get("name", "")).strip()
        prompt = str(spec.get("prompt", "")).strip()
        if not name or not prompt:
            raise ValueError("每个类别都需要导出名称和检测提示词")
        if "\n" in name or "\n" in prompt:
            raise ValueError("类别名称和提示词不能换行")
        if name in names:
            raise ValueError(f"类别名称重复: {name}")
        if prompt in prompts:
            raise ValueError(f"检测提示词重复: {prompt}")
        names.add(name)
        prompts.add(prompt)
        result.append({"name": name, "prompt": prompt})
    return result


def _polygon_from_mask(mask, box, height: int, width: int):
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((np.clip(x1, 0, width), np.clip(x2, 0, width)))
    y1, y2 = sorted((np.clip(y1, 0, height), np.clip(y2, 0, height)))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    points = None
    if mask is not None:
        binary = np.asarray(mask, dtype=np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        restricted = np.zeros_like(binary)
        xa, ya = int(np.floor(x1)), int(np.floor(y1))
        xb, yb = int(np.ceil(x2)), int(np.ceil(y2))
        restricted[ya:yb, xa:xb] = binary[ya:yb, xa:xb]
        points = cv2.findNonZero(restricted)
    if points is not None and len(points) >= 15:
        center, size, angle = cv2.minAreaRect(points)
        polygon = cv2.boxPoints((center, (size[0] * 1.04 + 2,
                                                size[1] * 1.04 + 2), angle))
    else:
        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon if cv2.contourArea(polygon) >= 25 else None


def _polygon_iou(first, second) -> float:
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def select_detections(image, roi, detections: list[dict], class_count: int,
                      count_mode: str, target_count: int) -> dict:
    """Normalize polygons, remove duplicate boxes, and apply the count policy."""
    if count_mode not in {"all", "fixed"}:
        raise ValueError(f"未知目标数量模式: {count_mode}")
    if count_mode == "fixed" and target_count < 1:
        raise ValueError("固定目标数至少为 1")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = roi_pixels(roi, width, height)
    crop_height, crop_width = y2 - y1, x2 - x1
    candidates = []
    for detection in detections:
        class_id = int(detection["class_id"])
        score = float(detection["score"])
        if not 0 <= class_id < class_count or not np.isfinite(score):
            continue
        polygon = _polygon_from_mask(
            detection.get("mask"), detection["box"], crop_height, crop_width)
        if polygon is None:
            continue
        polygon += np.array([x1, y1], dtype=np.float32)
        if not polygon_in_roi(polygon, roi, width, height):
            continue
        candidates.append({"class_id": class_id, "score": score,
                           "polygon": polygon})
    candidates.sort(key=lambda item: item["score"], reverse=True)

    unique = []
    ambiguous_class = False
    for candidate in candidates:
        duplicate = next((kept for kept in unique
                          if _polygon_iou(candidate["polygon"], kept["polygon"]) >= NMS_IOU),
                         None)
        if duplicate is not None:
            if (candidate["class_id"] != duplicate["class_id"] and
                    duplicate["score"] - candidate["score"] < 0.15):
                ambiguous_class = True
            continue
        unique.append(candidate)

    chosen = unique[:target_count] if count_mode == "fixed" else unique
    objects = [{"class_id": item["class_id"],
                "score": round(item["score"], 4),
                "polygon": item["polygon"].round(2).tolist()}
               for item in chosen]
    near_edge = any(
        np.any(item["polygon"][:, 0] <= x1 + 2) or
        np.any(item["polygon"][:, 0] >= x2 - 3) or
        np.any(item["polygon"][:, 1] <= y1 + 2) or
        np.any(item["polygon"][:, 1] >= y2 - 3)
        for item in chosen
    )
    if not objects:
        reason = "区域内未找到目标，请人工补框或确认无目标"
    elif count_mode == "fixed" and len(objects) < target_count:
        reason = f"只找到 {len(objects)} 个目标，少于设定的 {target_count} 个"
    elif count_mode == "fixed" and any(
            item["score"] >= EXTRA_SCORE for item in unique[target_count:]):
        reason = "高分候选超过设定数量，请人工选择"
    elif ambiguous_class:
        reason = "同一位置匹配多个类别，请人工核对类别"
    elif near_edge:
        reason = "目标贴近标注区域边缘，请人工检查"
    elif any(item["score"] < AUTO_SCORE for item in chosen):
        reason = "存在低置信度目标，请人工检查"
    else:
        reason = "自动候选，可人工复核"
    status = "auto" if reason == "自动候选，可人工复核" else "review"
    return {"objects": objects, "status": status, "reason": reason,
            "candidate_count": len(unique)}


def detect_generic(image, roi, segmenter, classes: list[dict],
                   count_mode: str, target_count: int) -> dict:
    x1, y1, x2, y2 = roi_pixels(roi, image.shape[1], image.shape[0])
    detections = segmenter.detect_many(
        image[y1:y2, x1:x2], [item["prompt"] for item in classes])
    return select_detections(image, roi, detections, len(classes),
                             count_mode, target_count)


def yolo_row(obj: dict, width: int, height: int, box_format: str,
             class_count: int) -> str:
    class_id = int(obj["class_id"])
    if not 0 <= class_id < class_count:
        raise ValueError(f"类别编号无效: {class_id}")
    polygon = np.asarray(obj["polygon"], dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("目标框无效")
    if box_format == "obb":
        values = (polygon / np.array([width, height], dtype=np.float32)).reshape(-1)
    elif box_format == "hbb":
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        values = [(lower[0] + upper[0]) / (2 * width),
                  (lower[1] + upper[1]) / (2 * height),
                  (upper[0] - lower[0]) / width,
                  (upper[1] - lower[1]) / height]
    else:
        raise ValueError(f"未知标签格式: {box_format}")
    if np.any(np.asarray(values) < 0) or np.any(np.asarray(values) > 1):
        raise ValueError("目标框超出图片边界")
    return f"{class_id} " + " ".join(f"{float(value):.6f}" for value in values) + "\n"


def export_generic(source_dir: Path, output_dir: Path, records: dict[str, dict],
                   classes: list[dict], box_format: str) -> tuple[Path, int, int]:
    classes = validate_classes(classes)
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
    annotated_images = 0
    object_count = 0
    for record in sorted(selected, key=lambda item: item["name"]):
        name = record["name"]
        source = source_dir / name
        if Path(name).name != name or not source.is_file():
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
        objects = record.get("objects", [])
        if objects:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"无法读取图片: {source}")
            height, width = image.shape[:2]
            label = "".join(yolo_row(obj, width, height, box_format, len(classes))
                            for obj in objects)
            annotated_images += 1
            object_count += len(objects)
        else:
            label = ""
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    names_yaml = "\n".join(
        f"  {index}: {json.dumps(spec['name'], ensure_ascii=False)}"
        for index, spec in enumerate(classes))
    (target / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(classes)}\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    return target, annotated_images, object_count
