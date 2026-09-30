"""Pure helpers for conservative OpenTouch right-hand bbox selection.

This module deliberately has no Qt, Torch, Ultralytics, or HDF5 imports.  It
keeps the identity/output contract and the side-verification policy easy to
unit test without loading the annotation application or a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Detection:
    """One semantic detector result in source-image pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class RightHandSelection:
    """Result of the conservative right-hand decision for one frame."""

    bbox_xyxy: tuple[float, float, float, float] | None
    reason: str


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""

    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def prepare_detections(
    boxes: Iterable[Sequence[float]],
    scores: Iterable[float] | None,
    image_width: int,
    image_height: int,
    duplicate_iou: float = 0.70,
) -> list[Detection]:
    """Clip, validate, sort, and de-duplicate semantic detections."""

    raw_boxes = list(boxes)
    raw_scores = list(scores) if scores is not None else []
    if not raw_scores:
        raw_scores = [1.0] * len(raw_boxes)
    elif len(raw_scores) < len(raw_boxes):
        raw_scores.extend([1.0] * (len(raw_boxes) - len(raw_scores)))

    candidates: list[Detection] = []
    for box, score in zip(raw_boxes, raw_scores):
        if len(box) != 4:
            continue
        values = [float(value) for value in box]
        if not all(math.isfinite(value) for value in values):
            continue
        x1 = min(max(values[0], 0.0), float(image_width))
        y1 = min(max(values[1], 0.0), float(image_height))
        x2 = min(max(values[2], 0.0), float(image_width))
        y2 = min(max(values[3], 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(Detection((x1, y1, x2, y2), float(score)))

    selected: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(
            bbox_iou(candidate.bbox_xyxy, previous.bbox_xyxy) >= duplicate_iou
            for previous in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _touches_frame_border(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    border_margin_ratio: float,
) -> bool:
    margin_x = max(1.0, float(image_width) * float(border_margin_ratio))
    margin_y = max(1.0, float(image_height) * float(border_margin_ratio))
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= float(image_width) - margin_x
        or y2 >= float(image_height) - margin_y
    )


def select_verified_right_hand(
    generic_hands: Sequence[Detection],
    right_hands: Sequence[Detection],
    left_hands: Sequence[Detection],
    image_width: int,
    image_height: int,
    match_iou: float = 0.45,
    side_score_margin: float = 0.05,
    border_margin_ratio: float = 0.002,
) -> RightHandSelection:
    """Select exactly one confidently identifiable, fully visible right hand.

    A right-prompt result must be confirmed by the generic black-gloved-hand
    prompt.  If a matching left-prompt result has a score too close to the
    right score, anatomical side is considered ambiguous.  Multiple plausible
    right-hand candidates are also ambiguous.  A candidate touching the image
    edge is treated as not fully visible.
    """

    if not generic_hands or not right_hands:
        return RightHandSelection(None, "no_right_hand")

    verified: list[tuple[Detection, Detection]] = []
    saw_border_candidate = False
    saw_side_conflict = False

    for right in right_hands:
        generic = max(
            generic_hands,
            key=lambda item: bbox_iou(right.bbox_xyxy, item.bbox_xyxy),
        )
        if bbox_iou(right.bbox_xyxy, generic.bbox_xyxy) < match_iou:
            continue

        if _touches_frame_border(
            generic.bbox_xyxy,
            image_width,
            image_height,
            border_margin_ratio,
        ):
            saw_border_candidate = True
            continue

        overlapping_left = [
            left
            for left in left_hands
            if bbox_iou(left.bbox_xyxy, generic.bbox_xyxy) >= match_iou
        ]
        if overlapping_left:
            strongest_left = max(overlapping_left, key=lambda item: item.score)
            if right.score < strongest_left.score + side_score_margin:
                saw_side_conflict = True
                continue

        verified.append((right, generic))

    if not verified:
        if saw_side_conflict:
            return RightHandSelection(None, "left_right_ambiguous")
        if saw_border_candidate:
            return RightHandSelection(None, "right_hand_not_fully_visible")
        return RightHandSelection(None, "right_hand_unconfirmed")

    verified.sort(key=lambda pair: pair[0].score, reverse=True)
    best_right, best_generic = verified[0]
    competing = [
        pair
        for pair in verified[1:]
        if pair[0].score >= best_right.score - side_score_margin
        and bbox_iou(pair[1].bbox_xyxy, best_generic.bbox_xyxy) < match_iou
    ]
    if competing:
        return RightHandSelection(None, "multiple_right_candidates")

    return RightHandSelection(best_generic.bbox_xyxy, "useful")


def make_sample_id(source_file: str, clip_id: str, frame_index: int) -> str:
    """Build the canonical ``source::clip::000000`` OpenTouch sample ID."""

    source_stem = Path(str(source_file)).stem
    clip_id = str(clip_id)
    if not source_stem or not clip_id:
        raise ValueError("source_file and clip_id must not be empty")
    if "::" in source_stem or "::" in clip_id:
        raise ValueError("source_file and clip_id must not contain '::'")
    frame_index = int(frame_index)
    if frame_index < 0:
        raise ValueError("source_frame_index must be non-negative")
    return f"{source_stem}::{clip_id}::{frame_index:06d}"


def expand_bbox_xyxy(
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.10,
) -> tuple[float, float, float, float]:
    """Expand an ``xyxy`` box on every side and clip it to the image.

    ``padding_ratio`` is relative to the original box dimension.  For example,
    ``0.10`` adds 10% of the original width on both the left and right, and 10%
    of the original height on both the top and bottom.
    """

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four values")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    padding_ratio = max(0.0, float(padding_ratio))
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive width and height")
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width), x2 + pad_x),
        min(float(image_height), y2 + pad_y),
    )


def make_bbox_record(
    source_file: str,
    clip_id: str,
    frame_index: int,
    bbox_xyxy: Sequence[float] | None,
) -> dict:
    """Create one JSON-serializable output row in the required field order."""

    bbox = None
    if bbox_xyxy is not None:
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        bbox = [round(float(value), 2) for value in bbox_xyxy]
    return {
        "sample_id": make_sample_id(source_file, clip_id, frame_index),
        "source_file": Path(str(source_file)).name,
        "clip_id": str(clip_id),
        "source_frame_index": int(frame_index),
        "bbox_xyxy": bbox,
    }
