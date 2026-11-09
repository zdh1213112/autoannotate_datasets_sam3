"""Multi-class, variable-count SAM3 annotation selection and YOLO export."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import cv2
import numpy as np

from .generic_region import (
    crop_region, polygon_in_region, region_bounds, region_near_edge,
)


AUTO_SCORE = 0.50
NMS_IOU = 0.60
NMS_SMALLER_COVERAGE = 0.80
REVIEW_RULES = (
    ("missing", "未检测到目标"),
    ("short_count", "固定数量不足"),
    ("class_conflict", "同位置类别冲突"),
    ("near_edge", "目标贴近区域边缘"),
    ("low_score", "低置信度目标"),
)
DEFAULT_REVIEW_REASONS = tuple(key for key, _ in REVIEW_RULES)
DEFAULT_SELECTION_RULE = "confidence"
SELECTION_RULES = ("confidence", "area")


def normalize_sample_percent(percent=None) -> int:
    if percent is None:
        return 100
    if isinstance(percent, bool) or not isinstance(percent, int) or not 1 <= percent <= 100:
        raise ValueError("自动标注图片比例必须是 1% 到 100% 的整数")
    return percent


def normalize_sample_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("随机抽样种子无效")
    return seed


def sample_image_files(files: list[Path], percent: int, seed: int) -> list[Path]:
    """Pick a seeded random subset of the current image list."""
    percent = normalize_sample_percent(percent)
    seed = normalize_sample_seed(seed)
    ordered = sorted(files, key=lambda path: path.name)
    count = (len(ordered) * percent + 99) // 100
    if count == len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, count)


def normalize_selection_rule(rule=None) -> str:
    if rule is None:
        return DEFAULT_SELECTION_RULE
    if rule not in SELECTION_RULES:
        raise ValueError(f"未知标签框选择方式: {rule}")
    return rule


def normalize_review_reasons(reasons=None) -> list[str]:
    if reasons is None:
        return list(DEFAULT_REVIEW_REASONS)
    if not isinstance(reasons, (list, tuple, set)):
        raise ValueError("待复核原因配置无效")
    unknown = set(reasons) - set(DEFAULT_REVIEW_REASONS)
    if unknown:
        raise ValueError(f"未知待复核原因: {', '.join(sorted(unknown))}")
    return [key for key in DEFAULT_REVIEW_REASONS if key in reasons]


def normalize_review_score(score=None) -> float:
    if score is None:
        return AUTO_SCORE
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("低置信度复核阈值无效") from exc
    if not np.isfinite(value) or not 0.05 <= value <= 0.95:
        raise ValueError("低置信度复核阈值必须在 0.05 到 0.95 之间")
    return value


def low_score_reason(score: float) -> str:
    return f"存在低置信度目标（低于 {score:.2f}），请人工检查"


def recheck_review_scores(records: dict[str, dict], score: float,
                          enabled: bool = True) -> dict[str, dict]:
    """Update automatic review states when only the low-score cutoff changes."""
    score = normalize_review_score(score)
    updated = {}
    for name, original in records.items():
        record = dict(original)
        status = record.get("status")
        reason = str(record.get("reason", ""))
        was_low_review = status == "review" and reason.startswith("存在低置信度目标")
        is_low = enabled and any(float(obj.get("score", 1.0)) < score
                                 for obj in record.get("objects", []))
        if status == "auto" and is_low:
            record.update(status="review", reason=low_score_reason(score))
        elif was_low_review:
            side_unresolved = (record.get("camera_id") in {"cam_2", "cam_3"}
                               and record.get("objects")
                               and not record.get("side_autoclassified"))
            if is_low:
                record["reason"] = low_score_reason(score)
            elif side_unresolved:
                record["reason"] = "侧视相机的左右手仍待人工确认"
            else:
                record.update(status="auto", reason="自动候选，可人工复核")
        updated[name] = record
    return updated


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


def _polygons_duplicate(first, second) -> bool:
    intersection, _ = cv2.intersectConvexConvex(first, second)
    first_area, second_area = cv2.contourArea(first), cv2.contourArea(second)
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    return (union > 0 and intersection / union >= NMS_IOU or
            smaller > 0 and intersection / smaller >= NMS_SMALLER_COVERAGE)


def has_duplicate_selected_boxes(record: dict) -> bool:
    """Identify old auto results that should be reselected after NMS changes."""
    polygons = []
    for obj in record.get("objects", []):
        polygon = np.asarray(obj.get("polygon", []), dtype=np.float32)
        if polygon.shape != (4, 2):
            continue
        if any(_polygons_duplicate(polygon, previous) for previous in polygons):
            return True
        polygons.append(polygon)
    return False


def select_detections(image, roi, detections: list[dict], class_count: int,
                      count_mode: str, target_count: int,
                      review_reasons=None, selection_rule=None,
                      review_score=None) -> dict:
    """Keep every in-region prediction and select deduplicated output labels."""
    if count_mode not in {"all", "fixed"}:
        raise ValueError(f"未知目标数量模式: {count_mode}")
    if count_mode == "fixed" and target_count < 1:
        raise ValueError("固定目标数至少为 1")
    selection_rule = normalize_selection_rule(selection_rule)
    review_score = normalize_review_score(review_score)
    enabled_reasons = set(normalize_review_reasons(review_reasons))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region_bounds(roi, width, height)
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
        if not polygon_in_region(polygon, roi, width, height):
            continue
        candidates.append({"class_id": class_id, "score": score,
                           "polygon": polygon})
    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Keep candidate IDs in confidence order for the preview, while choosing
    # an NMS representative according to the selected fixed-count rule.
    ranked = list(enumerate(candidates))
    if count_mode == "fixed" and selection_rule == "area":
        ranked.sort(key=lambda pair: (cv2.contourArea(pair[1]["polygon"]),
                                       pair[1]["score"]), reverse=True)
    unique = []
    ambiguous_class = False
    for index, candidate in ranked:
        duplicate = next((kept for _, kept in unique
                          if _polygons_duplicate(candidate["polygon"],
                                                 kept["polygon"])),
                         None)
        if duplicate is not None:
            if (candidate["class_id"] != duplicate["class_id"] and
                    abs(duplicate["score"] - candidate["score"]) < 0.15):
                ambiguous_class = True
            continue
        unique.append((index, candidate))

    chosen = unique[:target_count] if count_mode == "fixed" else unique
    chosen_ids = {index for index, _ in chosen}
    objects = [{"class_id": item["class_id"],
                "score": round(item["score"], 4),
                "polygon": item["polygon"].round(2).tolist(),
                "candidate_id": index}
               for index, item in chosen]
    visible_candidates = [
        {"class_id": item["class_id"], "score": round(item["score"], 4),
         "polygon": item["polygon"].round(2).tolist(),
         "selected": index in chosen_ids}
        for index, item in enumerate(candidates)
    ]
    near_edge = any(region_near_edge(item["polygon"], roi, width, height)
                    for _, item in chosen)
    triggered = []
    if not objects:
        triggered.append(("missing", "区域内未找到目标，请人工补框或确认无目标"))
    if count_mode == "fixed" and len(objects) < target_count:
        triggered.append(("short_count", f"只找到 {len(objects)} 个目标，少于设定的 {target_count} 个"))
    if ambiguous_class:
        triggered.append(("class_conflict", "同一位置匹配多个类别，请人工核对类别"))
    if near_edge:
        triggered.append(("near_edge", "目标贴近标注区域边缘，请人工检查"))
    if any(item["score"] < review_score for _, item in chosen):
        triggered.append(("low_score", low_score_reason(review_score)))
    selected = next((text for key, text in triggered if key in enabled_reasons), None)
    if selected:
        status, reason = "review", selected
    elif triggered:
        status, reason = "auto", f"{triggered[0][1]}（该复核条件已关闭）"
    else:
        status, reason = "auto", "自动候选，可人工复核"
    return {"objects": objects, "candidates": visible_candidates,
            "status": status, "reason": reason,
            "candidate_count": len(visible_candidates)}


def sync_candidates(record: dict) -> dict:
    """Match selected objects to display candidates after manual editing."""
    candidates = [dict(item) for item in record.get("candidates", [])]
    objects = []
    selected_ids = set()
    for source in record.get("objects", []):
        obj = dict(source)
        candidate_id = obj.get("candidate_id")
        if (not isinstance(candidate_id, int) or isinstance(candidate_id, bool)
                or candidate_id < 0 or candidate_id >= len(candidates)
                or candidate_id in selected_ids):
            candidate_id = len(candidates)
            obj["candidate_id"] = candidate_id
            candidates.append({})
        selected_ids.add(candidate_id)
        candidates[candidate_id].update(
            class_id=obj["class_id"], score=obj.get("score", 1.0),
            polygon=obj["polygon"], selected=True)
        objects.append(obj)
    for index, item in enumerate(candidates):
        item["selected"] = index in selected_ids
    return {**record, "objects": objects, "candidates": candidates,
            "candidate_count": max(int(record.get("candidate_count", 0)), len(candidates))}


def detect_generic(image, roi, segmenter, classes: list[dict],
                   count_mode: str, target_count: int,
                   review_reasons=None, selection_rule=None,
                   review_score=None) -> dict:
    detections = segmenter.detect_many(
        crop_region(image, roi), [item["prompt"] for item in classes])
    return select_detections(image, roi, detections, len(classes),
                             count_mode, target_count, review_reasons,
                             selection_rule, review_score)


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
                   classes: list[dict], box_format: str,
                   allowed_names: set[str] | None = None) -> tuple[Path, int, int]:
    classes = validate_classes(classes)
    selected = [record for record in records.values()
                if record["status"] in {"auto", "confirmed", "empty_confirmed"}
                and (allowed_names is None or record["name"] in allowed_names)]
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
