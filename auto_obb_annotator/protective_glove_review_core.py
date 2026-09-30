"""Core selection, side assignment and export helpers for glove review mode.

This module deliberately contains no Qt or model imports.  It keeps the
left/right policy and the review JSON contract testable without starting the
GUI or loading SAM3.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from .generic_core import yolo_row
from .protective_glove_core import (
    ProtectiveGloveFilterConfig,
    mask_for_image,
    select_protective_gloved_hands,
)


CLASS_NAMES = ("left", "right")
SINGLE_HAND_REASON = "单手左右自动证据不足；按 1 指定左手、2 指定右手并跳到下一张"
SIDE_SCORE_MARGIN = 0.08
AUTO_SIDE_SOURCES = {"manual", "semantic", "temporal"}
STATUS_TEXT = {
    "auto": "自动候选",
    "review": "待人工复核",
    "confirmed": "人工确认",
    "empty_confirmed": "人工确认无目标",
}


def _as_list(value):
    """Convert model outputs to a list without triggering ndarray truth tests."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value)


def _box_polygon(box, width: int, height: int) -> np.ndarray | None:
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, x2 = sorted((np.clip(x1, 0.0, width - 1.0),
                     np.clip(x2, 0.0, width - 1.0)))
    y1, y2 = sorted((np.clip(y1, 0.0, height - 1.0),
                     np.clip(y2, 0.0, height - 1.0)))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                      dtype=np.float32)


def polygon_from_mask(mask, fallback_box, width: int, height: int) -> list[list[float]] | None:
    """Fit one four-corner OBB to a completed mask, with an HBB fallback."""

    try:
        binary = mask_for_image(mask, width, height)
    except (TypeError, ValueError):
        binary = None
    polygon = None
    if binary is not None:
        points = cv2.findNonZero(binary.astype(np.uint8))
        if points is not None and len(points) >= 4:
            rect = cv2.minAreaRect(points)
            candidate = cv2.boxPoints(rect)
            if cv2.contourArea(candidate) >= 25.0:
                polygon = candidate
    if polygon is None:
        polygon = _box_polygon(fallback_box, width, height)
    if polygon is None:
        return None
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, width - 1.0)
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, height - 1.0)
    return polygon.round(2).tolist()


def _center_x(obj: dict) -> float:
    polygon = np.asarray(obj["polygon"], dtype=np.float32).reshape(4, 2)
    return float(polygon[:, 0].mean())


def _side_name(class_id: int | None) -> str:
    return CLASS_NAMES[class_id] if class_id in (0, 1) else "unknown"


def resolved_sides(objects: list[dict]) -> bool:
    """Unknown is a review state, never a third training class."""
    sides = [obj.get("class_id") for obj in objects]
    return (0 < len(sides) <= 2 and all(side in (0, 1) for side in sides)
            and len(set(sides)) == len(sides))


def normalize_record_sides(record: dict) -> dict:
    """Remove legacy positional guesses on unconfirmed single-hand records.

    Run after reading the latest JSONL row, without rewriting the source log.
    Previously confirmed labels remain the human's decision. New records carry
    provenance so a single hand explicitly assigned by a person stays assigned.
    """
    record = {**record, "objects": [dict(obj) for obj in record.get("objects", [])]}
    objects = record["objects"]
    for obj in objects:
        if obj.get("class_id") not in (0, 1):
            obj.update(class_id=None, side="unknown", side_source="unresolved")
        else:
            obj["side"] = _side_name(obj["class_id"])
            if record.get("status") == "confirmed":
                obj["side_source"] = "manual"
    if (len(objects) == 1 and record.get("status") in {"auto", "review"}
            and objects[0].get("side_source") not in AUTO_SIDE_SOURCES):
        objects[0].update(class_id=None, side="unknown", side_source="unresolved")
        record.update(status="review", reason=SINGLE_HAND_REASON)
    if objects and not resolved_sides(objects):
        reason = SINGLE_HAND_REASON if len(objects) == 1 else "请明确每个框的左右类别，且左右各最多一个框"
        record.update(status="review", reason=reason)
    return record


def assign_left_right_classes(
    objects: list[dict],
    image_width: int,
    side_mapping: str = "screen",
) -> list[dict]:
    """Assign classes by horizontal position, independent of model order.

    ``screen`` means the image-left glove is class ``left`` and image-right is
    class ``right``.  ``mirror`` swaps the two class IDs for mirrored cameras.
    A single glove has no positional handedness evidence. Preserve a manual,
    semantic or temporal identity, otherwise leave it unknown regardless of x.
    """

    if side_mapping not in {"screen", "mirror"}:
        raise ValueError("side_mapping must be 'screen' or 'mirror'")
    if not objects:
        return objects
    ordered = sorted(objects, key=_center_x)
    if len(ordered) == 1:
        obj = ordered[0]
        if (obj.get("side_source") not in AUTO_SIDE_SOURCES
                or obj.get("class_id") not in (0, 1)):
            obj.update(class_id=None, side="unknown", side_source="unresolved")
        return ordered
    # The first two are the only valid automatic hand slots.  A caller may
    # retain extra candidates for diagnostics, but they must not be exported.
    for index, obj in enumerate(ordered):
        class_id = 0 if index == 0 else 1
        if side_mapping == "mirror":
            class_id = 1 - class_id
        obj.update(class_id=class_id, side=_side_name(class_id), side_source="position_pair")
    return ordered


def _side_reason(objects: list[dict], image_width: int, side_mapping: str) -> str:
    if not objects:
        return "未找到手套，请人工补框或确认无目标"
    if len(objects) == 1:
        return SINGLE_HAND_REASON
    return "自动按画面左右分配 left/right，可人工复核"


def _prompt_side(prompt: str) -> int | None:
    """Return the anatomical side named by a prompt, if it names one."""

    words = {word.lower() for word in str(prompt).replace("-", " ").split()}
    if "left" in words:
        return 0
    if "right" in words:
        return 1
    return None


def _semantic_side(obj: dict, auto_score: float,
                   margin: float = SIDE_SCORE_MARGIN) -> int | None:
    """Choose a single-hand side only when SAM3 gives separated evidence."""

    side_scores = obj.get("side_scores") or {}
    scores = {
        side: float(side_scores.get(side, side_scores.get(str(side), 0.0)) or 0.0)
        for side in (0, 1)
    }
    if not all(np.isfinite(score) for score in scores.values()):
        return None
    best, other = max(scores, key=scores.get), min(scores, key=scores.get)
    if scores[best] < auto_score or scores[best] - scores[other] < margin:
        return None
    return best


def _polygon_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    overlap, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - overlap
    return float(overlap / union) if union > 0 else 0.0


def _review_decision(objects, candidate_count, width, side_mapping, auto_score):
    """Judge box quality separately from the wording of the detection prompt.

    Paired hands follow screen/mirror convention; single hands need independent
    side evidence. Detection scores are not calibrated handedness probabilities.
    Border contact is normal in egocentric images.
    """
    if not objects:
        return "review", "未找到手套，请人工补框或确认无目标"
    if candidate_count > 2:
        return "review", f"检测到 {candidate_count} 个独立候选，只保留左右两只，请人工选择"
    if len(objects) == 1:
        obj = objects[0]
        if (obj.get("class_id") in (0, 1)
                and obj.get("side_source") in AUTO_SIDE_SOURCES):
            if float(obj.get("score", 0.0)) < auto_score:
                return "review", "单手左右已有线索，但手套框置信度偏低，请人工检查"
            source_text = {
                "semantic": "SAM3 左右手语义提示词已明确单手类别",
                "temporal": "已根据相邻帧的已知左右手身份延续单手类别",
                "manual": "已人工指定单手类别",
            }.get(obj.get("side_source"), "单手类别已明确")
            return "auto", source_text
        return "review", _side_reason(objects, width, side_mapping)
    if (_polygon_iou(objects[0]["polygon"], objects[1]["polygon"]) >= 0.15
            or abs(_center_x(objects[0]) - _center_x(objects[1])) < width * 0.05):
        return "review", "两只手套靠近或交叠，左右位置不明确，请人工检查"
    if any(float(obj["score"]) < auto_score for obj in objects):
        return "review", "存在低置信度手套，请人工检查"
    return "auto", "两个高置信度独立手套框，已按画面左右规则分配类别"


def make_glove_record(
    name: str,
    width: int,
    height: int,
    objects: list[dict],
    status: str,
    reason: str,
    candidate_count: int | None = None,
) -> dict:
    """Normalize one review record to a JSON-serializable structure."""

    normalized = []
    for obj in objects:
        polygon = np.asarray(obj.get("polygon"), dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            continue
        class_id = obj.get("class_id")
        if class_id not in (0, 1):
            class_id = None
        normalized_obj = {
            "class_id": class_id,
            "side": _side_name(class_id),
            "side_source": obj.get("side_source", "unresolved"),
            "score": round(float(obj.get("score", 0.0)), 4),
            "polygon": polygon.round(2).tolist(),
        }
        side_scores = obj.get("side_scores")
        if isinstance(side_scores, dict):
            normalized_obj["side_scores"] = {
                str(side): round(float(score), 4)
                for side, score in side_scores.items()
                if side in (0, 1, "0", "1") and np.isfinite(float(score))
            }
        normalized.append(normalized_obj)
    return normalize_record_sides({
        "name": str(name),
        "width": int(width),
        "height": int(height),
        "objects": normalized,
        "status": str(status),
        "reason": str(reason),
        "candidate_count": int(candidate_count if candidate_count is not None else len(normalized)),
    })


def select_glove_objects(
    image_bgr,
    masks,
    boxes,
    scores,
    config: ProtectiveGloveFilterConfig = ProtectiveGloveFilterConfig(
        complete_box_enabled=True,
    ),
    side_mapping: str = "screen",
    auto_score: float = 0.70,
) -> dict:
    """Convert SAM3 glove candidates into a review record payload."""

    if image_bgr is None or image_bgr.size == 0:
        return {"objects": [], "status": "review", "reason": "图片为空",
                "candidate_count": 0}
    height, width = image_bgr.shape[:2]
    # Request all reasonable candidates here.  The UI may still be configured
    # to keep at most two for export, but seeing the count lets review explain
    # why an extra glove was suppressed.
    all_config = ProtectiveGloveFilterConfig(
        max_value=config.max_value,
        min_dark_ratio=config.min_dark_ratio,
        min_box_area_ratio=config.min_box_area_ratio,
        min_center_y_ratio=config.min_center_y_ratio,
        max_hands=max(2, int(config.max_hands)),
        box_padding_ratio=config.box_padding_ratio,
        fragment_gap_ratio=config.fragment_gap_ratio,
        complete_box_enabled=True,
    )
    selected_masks, selected_boxes, selected_scores = select_protective_gloved_hands(
        image_bgr, _as_list(masks), _as_list(boxes), _as_list(scores), all_config)
    candidates = []
    for mask, box, score in zip(selected_masks, selected_boxes, selected_scores):
        polygon = polygon_from_mask(mask, box, width, height)
        if polygon is None:
            continue
        candidates.append({"class_id": None, "score": float(score), "polygon": polygon})
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    candidate_count = len(candidates)
    objects = assign_left_right_classes(candidates[:2], width, side_mapping)
    status, reason = _review_decision(
        objects, candidate_count, width, side_mapping, auto_score)
    return {"objects": objects, "status": status, "reason": reason,
            "candidate_count": candidate_count}


def select_prompted_glove_objects(
    image_bgr,
    prompt_results: dict[str, tuple],
    config: ProtectiveGloveFilterConfig = ProtectiveGloveFilterConfig(
        complete_box_enabled=True,
    ),
    side_mapping: str = "screen",
    auto_score: float = 0.70,
) -> dict:
    """Merge prompt results, preserving semantic evidence for single hands.

    The segmenter applies ``config`` before returning these masks. A left/right
    prompt can find both hands, so only a strong, separated score resolves a
    single hand. Paired hands retain the selected screen/mirror convention.
    """

    if image_bgr is None or image_bgr.size == 0:
        return {"objects": [], "status": "review", "reason": "图片为空",
                "candidate_count": 0}
    height, width = image_bgr.shape[:2]
    normalized_results = prompt_results or {}
    candidates = []
    for prompt, result in normalized_results.items():
        try:
            masks, boxes, scores = result
        except (TypeError, ValueError):
            continue
        masks = _as_list(masks)
        boxes = _as_list(boxes)
        scores = _as_list(scores)
        if len(scores) < len(boxes):
            scores.extend([0.0] * (len(boxes) - len(scores)))
        for index, (box, score) in enumerate(zip(boxes, scores)):
            mask = masks[index] if index < len(masks) else None
            score = float(score)
            if not np.isfinite(score):
                continue
            polygon = polygon_from_mask(mask, box, width, height)
            if polygon is None or not polygon_valid(polygon, width, height):
                continue
            candidates.append({
                "class_id": None,
                "score": score,
                "polygon": polygon,
                "prompt_side": _prompt_side(prompt),
            })

    # One physical glove can appear in all three prompt outputs. Merge those
    # detections spatially, while retaining the strongest left/right semantic
    # score for the cluster.  The old implementation discarded this evidence
    # and consequently had to send every single hand to manual review.
    clusters = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        cluster = next((item for item in clusters
                        if _polygon_iou(candidate["polygon"], item["polygon"]) >= 0.60),
                       None)
        if cluster is None:
            cluster = {
                "class_id": None,
                "score": float(candidate["score"]),
                "polygon": candidate["polygon"],
                "side_scores": {},
            }
            clusters.append(cluster)
        else:
            cluster["score"] = max(float(cluster["score"]),
                                   float(candidate["score"]))
        prompt_side = candidate.get("prompt_side")
        if prompt_side in (0, 1):
            side_scores = cluster.setdefault("side_scores", {})
            side_scores[prompt_side] = max(
                float(side_scores.get(prompt_side, 0.0)),
                float(candidate["score"]),
            )
    for cluster in clusters:
        side = _semantic_side(cluster, auto_score)
        if side is not None:
            cluster.update(class_id=side, side=_side_name(side),
                           side_source="semantic")
    candidate_count = len(clusters)
    objects = [dict(item) for item in clusters[:2]]
    objects = assign_left_right_classes(objects, width, side_mapping)
    status, reason = _review_decision(
        objects, candidate_count, width, side_mapping, auto_score)
    return {"objects": objects, "status": status, "reason": reason,
            "candidate_count": candidate_count}


def apply_temporal_side(payload: dict, previous_objects: list[dict] | None,
                        image_width: int, image_height: int,
                        auto_score: float = 0.70) -> dict:
    """Use the immediately preceding frame to resolve an ambiguous single hand.

    A semantic side from the current frame always wins.  Temporal matching is
    deliberately limited to one unresolved candidate and a nearby previous
    box, so a stale identity cannot silently label an unrelated image.
    """

    result = {**payload, "objects": [dict(obj) for obj in payload.get("objects", [])]}
    objects = result["objects"]
    if (len(objects) != 1 or result.get("candidate_count", len(objects)) != 1
            or not previous_objects):
        return result
    current = objects[0]
    if current.get("class_id") in (0, 1) and current.get("side_source") in AUTO_SIDE_SOURCES:
        return result
    try:
        current_points = np.asarray(current["polygon"], dtype=np.float32)
        current_center = current_points.reshape(4, 2).mean(axis=0)
        diagonal = max(1.0, float(np.hypot(image_width, image_height)))
    except (KeyError, TypeError, ValueError):
        return result
    if not polygon_valid(current_points, image_width, image_height):
        return result
    matches = []
    for previous in previous_objects:
        if previous.get("class_id") not in (0, 1):
            continue
        try:
            previous_points = np.asarray(previous["polygon"], dtype=np.float32)
            if not polygon_valid(previous_points, image_width, image_height):
                continue
            previous_center = previous_points.reshape(4, 2).mean(axis=0)
            area_ratio = cv2.contourArea(current_points) / cv2.contourArea(previous_points)
            if not 0.33 <= area_ratio <= 3.0:
                continue
            iou = _polygon_iou(current_points, previous_points)
            distance = float(np.linalg.norm(current_center - previous_center) / diagonal)
        except (KeyError, TypeError, ValueError, cv2.error):
            continue
        if iou < 0.15 and distance > 0.20:
            continue
        affinity = max(iou / 0.35, (0.20 - distance) / 0.20)
        matches.append((affinity, iou, distance, previous))
    if not matches:
        return result
    matches.sort(key=lambda item: item[0], reverse=True)
    best = matches[0]
    if any(other[3]["class_id"] != best[3]["class_id"]
           and best[0] - other[0] < 0.20 for other in matches[1:]):
        return result
    previous = best[3]
    side = int(previous["class_id"])
    current.update(class_id=side, side=_side_name(side), side_source="temporal")
    current["temporal_iou"] = round(best[1], 4)
    current["temporal_distance"] = round(best[2], 4)
    if float(current.get("score", 0.0)) >= auto_score:
        result.update(status="auto",
                      reason="当前单手语义不充分，已由相邻帧的已知左右手身份自动延续")
    return result


def temporal_seed_objects(previous: dict | None, name: str,
                          width: int, height: int) -> list[dict]:
    """Only trust adjacent numbered frames of the same sequence and size."""
    if not previous or previous.get("status") not in {"auto", "confirmed"}:
        return []
    previous = normalize_record_sides(previous)
    if (previous.get("status") not in {"auto", "confirmed"}
            or (previous.get("width"), previous.get("height")) != (width, height)
            or not resolved_sides(previous.get("objects", []))):
        return []
    old = re.fullmatch(r"(.*?)(\d+)(\D*)", previous.get("name", ""))
    new = re.fullmatch(r"(.*?)(\d+)(\D*)", name)
    if (not old or not new or old.group(1, 3) != new.group(1, 3)
            or int(new.group(2)) != int(old.group(2)) + 1):
        return []
    return [dict(obj) for obj in previous["objects"]]


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if record.get("name"):
                    records[record["name"]] = record
    return {name: normalize_record_sides(record) for name, record in records.items()}


def image_files(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in suffixes)


def polygon_valid(polygon, width: int, height: int) -> bool:
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if cv2.contourArea(points) < 25:
        return False
    return bool(np.all(points >= 0) and np.all(points[:, 0] <= width)
                and np.all(points[:, 1] <= height))


def export_dataset(source_dir: Path, output_dir: Path, records: dict[str, dict],
                   box_format: str) -> tuple[Path, int, int]:
    """Export confirmed glove records and omit unresolved review rows."""

    if box_format not in {"obb", "hbb"}:
        raise ValueError(f"未知标签格式: {box_format}")
    selected = [record for record in records.values()
                if record.get("status") in {"auto", "confirmed", "empty_confirmed"}]
    # Fail before creating an export, instead of dropping an unresolved glove
    # or accidentally coercing its class to left (0).
    for record in selected:
        objects = record.get("objects", [])
        if record["status"] == "empty_confirmed":
            if objects:
                raise ValueError(f"{record['name']}: 无目标记录仍含手套框")
        elif not resolved_sides(objects):
            raise ValueError(f"{record['name']}: 左右类别未明确，无法导出，请先人工指定")
        elif (len(objects) == 1 and record["status"] == "auto"
              and objects[0].get("side_source") not in AUTO_SIDE_SOURCES):
            raise ValueError(f"{record['name']}: 单手左右证据不足，无法导出")
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
        objects = []
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        height, width = image.shape[:2]
        for obj in record.get("objects", []):
            if polygon_valid(obj.get("polygon"), width, height):
                objects.append(obj)
        label = "".join(yolo_row(obj, width, height, box_format, len(CLASS_NAMES))
                        for obj in objects)
        if objects:
            annotated_images += 1
            object_count += len(objects)
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    names_yaml = "\n".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}"
                           for index, name in enumerate(CLASS_NAMES))
    (target / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names_yaml}\n",
        encoding="utf-8")
    (target / "README.txt").write_text(
        "Protective-glove dataset. Class 0 is left and class 1 is right.\n"
        "Rows still marked review were omitted from the export.\n",
        encoding="utf-8")
    return target, annotated_images, object_count
