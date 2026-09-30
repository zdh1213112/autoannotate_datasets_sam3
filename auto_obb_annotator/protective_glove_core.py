"""Pure image helpers for complete mixed-color protective-glove boxes.

The helpers in this module deliberately avoid Qt, Torch and Ultralytics so the
mask completion policy can be tested without loading the annotation UI or a
large model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ProtectiveGloveFilterConfig:
    """Thresholds used after SAM3 semantic segmentation."""

    max_value: int = 170
    # Zero disables the optional dark-component check.  The target glove has
    # white/gray armor as well as black fingers, so color must not be a gate by
    # default.
    min_dark_ratio: float = 0.0
    min_box_area_ratio: float = 0.002
    min_center_y_ratio: float = 0.15
    max_hands: int = 2
    box_padding_ratio: float = 0.08
    fragment_gap_ratio: float = 0.04
    # Opt-in keeps other launchers that reuse this selector byte-for-byte
    # compatible with their original mask/box behavior.
    complete_box_enabled: bool = False


def mask_for_image(mask, img_w: int, img_h: int) -> np.ndarray:
    """Return a two-dimensional boolean mask at source-image resolution."""

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("mask must become two-dimensional after squeeze")
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(
            mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def complete_glove_mask(
    mask_bool: np.ndarray,
    semantic_box,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.08,
    fragment_gap_ratio: float = 0.04,
) -> tuple[np.ndarray, list[float]]:
    """Build one connected mask and a padded box for the whole glove.

    SAM masks may contain separate finger islands, while the generic exporter
    uses only the largest contour.  This function closes small breaks, removes
    distant speckles, takes the hull of the remaining instance together with
    SAM's semantic box, and dilates the hull to add a configurable safety
    margin.  The returned mask therefore produces complete HBB and OBB labels
    in the existing export pipeline.
    """

    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    if mask_u8.shape != (image_height, image_width):
        mask_u8 = cv2.resize(
            mask_u8, (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return mask_u8.astype(bool), [float(value) for value in semantic_box]

    raw_x1, raw_y1, raw_x2, raw_y2 = [
        float(value) for value in semantic_box]
    x1, x2 = sorted((raw_x1, raw_x2))
    y1, y2 = sorted((raw_y1, raw_y2))
    x1 = float(np.clip(x1, 0.0, image_width))
    x2 = float(np.clip(x2, 0.0, image_width))
    y1 = float(np.clip(y1, 0.0, image_height))
    y2 = float(np.clip(y2, 0.0, image_height))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    gap_ratio = max(0.0, float(fragment_gap_ratio))
    gap_x = max(1, int(round(box_w * gap_ratio)))
    gap_y = max(1, int(round(box_h * gap_ratio)))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_x * 2 + 1, gap_y * 2 + 1))
    joined = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    # Keep sizeable components close to the SAM instance box.  This rejects
    # isolated mask noise before the convex hull is constructed.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    cleaned = np.zeros_like(joined)
    if count > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(component_areas.max())
        min_component_area = max(8, int(round(largest_area * 0.01)))
        guard = (
            max(0.0, x1 - gap_x * 2),
            max(0.0, y1 - gap_y * 2),
            min(float(image_width), x2 + gap_x * 2),
            min(float(image_height), y2 + gap_y * 2),
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue
            cx = stats[component_id, cv2.CC_STAT_LEFT]
            cy = stats[component_id, cv2.CC_STAT_TOP]
            cw = stats[component_id, cv2.CC_STAT_WIDTH]
            ch = stats[component_id, cv2.CC_STAT_HEIGHT]
            intersects_guard = (
                cx + cw >= guard[0] and cy + ch >= guard[1]
                and cx <= guard[2] and cy <= guard[3]
            )
            if intersects_guard:
                cleaned[labels == component_id] = 1
    if not np.any(cleaned):
        cleaned = mask_u8

    ys, xs = np.nonzero(cleaned)
    points = np.column_stack((xs, ys)).astype(np.int32)
    semantic_corners = np.array([
        [int(round(x1)), int(round(y1))],
        [int(round(max(x1, x2 - 1))), int(round(y1))],
        [int(round(max(x1, x2 - 1))), int(round(max(y1, y2 - 1)))],
        [int(round(x1)), int(round(max(y1, y2 - 1)))],
    ], dtype=np.int32)
    hull = cv2.convexHull(np.vstack((points, semantic_corners)))
    completed = np.zeros_like(cleaned)
    cv2.fillConvexPoly(completed, hull, 1)

    padding_ratio = max(0.0, float(padding_ratio))
    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))
    if pad_x > 0 or pad_y > 0:
        padding_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, pad_x * 2 + 1), max(1, pad_y * 2 + 1)),
        )
        completed = cv2.dilate(completed, padding_kernel, iterations=1)

    ys, xs = np.nonzero(completed)
    complete_box = [
        float(xs.min()),
        float(ys.min()),
        float(min(image_width, int(xs.max()) + 1)),
        float(min(image_height, int(ys.max()) + 1)),
    ]
    return completed.astype(bool), complete_box


def select_protective_gloved_hands(
    image_bgr,
    masks,
    boxes,
    scores,
    config: ProtectiveGloveFilterConfig = ProtectiveGloveFilterConfig(),
):
    """Return complete masks/boxes for at most the best protective gloves."""

    if image_bgr is None or image_bgr.size == 0:
        return [], [], []

    img_h, img_w = image_bgr.shape[:2]
    image_area = float(max(1, img_w * img_h))
    value_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates = []

    score_values = list(scores) if scores is not None else []
    if not score_values:
        score_values = [1.0] * len(boxes)
    elif len(score_values) < len(boxes):
        score_values.extend([1.0] * (len(boxes) - len(score_values)))

    for index, (mask, box, score) in enumerate(zip(masks, boxes, score_values)):
        if len(box) != 4:
            continue
        values = [float(value) for value in box]
        if not all(math.isfinite(value) for value in values):
            continue
        x1 = float(np.clip(min(values[0], values[2]), 0.0, img_w))
        y1 = float(np.clip(min(values[1], values[3]), 0.0, img_h))
        x2 = float(np.clip(max(values[0], values[2]), 0.0, img_w))
        y2 = float(np.clip(max(values[1], values[3]), 0.0, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        box_area_ratio = (x2 - x1) * (y2 - y1) / image_area
        center_y_ratio = (y1 + y2) / (2.0 * img_h)

        try:
            mask_bool = mask_for_image(mask, img_w, img_h)
        except ValueError:
            continue
        mask_pixels = int(np.count_nonzero(mask_bool))
        if mask_pixels == 0:
            continue

        # Erosion avoids counting a thin halo introduced by mask resizing.
        mask_u8 = mask_bool.astype(np.uint8)
        inner = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        if np.count_nonzero(inner) < max(20, int(mask_pixels * 0.35)):
            inner = mask_bool
        dark_ratio = float(np.mean(value_channel[inner] <= config.max_value))

        if dark_ratio < config.min_dark_ratio:
            continue
        if box_area_ratio < config.min_box_area_ratio:
            continue
        if center_y_ratio < config.min_center_y_ratio:
            continue

        if config.complete_box_enabled:
            complete_mask, complete_box = complete_glove_mask(
                mask_bool,
                [x1, y1, x2, y2],
                img_w,
                img_h,
                padding_ratio=config.box_padding_ratio,
                fragment_gap_ratio=config.fragment_gap_ratio,
            )
        else:
            complete_mask = mask_bool
            complete_box = [x1, y1, x2, y2]
        area_quality = min(1.0, math.sqrt(box_area_ratio / 0.03))
        color_quality = dark_ratio if config.min_dark_ratio > 0.0 else 0.0
        quality = (
            float(score)
            + 0.75 * color_quality
            + 0.35 * area_quality
            + 0.25 * center_y_ratio
        )
        candidates.append({
            "index": index,
            "box": complete_box,
            "mask": complete_mask,
            "score": float(score),
            "quality": quality,
        })

    selected = []
    for candidate in sorted(candidates, key=lambda item: item["quality"], reverse=True):
        duplicate = any(
            _box_iou(candidate["box"], previous["box"]) >= 0.65
            or _mask_iou(candidate["mask"], previous["mask"]) >= 0.60
            for previous in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= max(1, int(config.max_hands)):
            break

    selected.sort(key=lambda item: item["index"])
    return (
        [item["mask"].astype(np.uint8) for item in selected],
        [item["box"] for item in selected],
        [item["score"] for item in selected],
    )
