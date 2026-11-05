"""Rectangle and circle search regions for the generic annotator."""

import cv2
import numpy as np

from .barcode_core import normalize_roi, polygon_in_roi, roi_pixels


FULL_REGION = [0.0, 0.0, 1.0, 1.0]


def normalize_region(region):
    """Keep legacy rectangle lists and validate new circle dictionaries."""
    if isinstance(region, dict):
        if region.get("shape") != "circle":
            return None
        try:
            center = [float(value) for value in region["center"]]
            radius = float(region["radius"])
        except (KeyError, TypeError, ValueError):
            return None
        if (len(center) != 2 or not np.isfinite(center + [radius]).all()
                or not all(0 <= value <= 1 for value in center)
                or not 0.005 <= radius <= 1):
            return None
        return {"shape": "circle", "center": center, "radius": radius}
    return normalize_roi(region)


def circle_pixels(region, width: int, height: int):
    region = normalize_region(region)
    if not isinstance(region, dict):
        raise ValueError("检测区域不是圆形")
    cx = region["center"][0] * width
    cy = region["center"][1] * height
    radius = region["radius"] * min(width, height)
    return cx, cy, radius


def region_bounds(region, width: int, height: int):
    region = normalize_region(region)
    if region is None:
        raise ValueError("请先选择有效的检测区域")
    if not isinstance(region, dict):
        return roi_pixels(region, width, height)
    cx, cy, radius = circle_pixels(region, width, height)
    x1 = max(0, min(width - 1, int(np.floor(cx - radius))))
    y1 = max(0, min(height - 1, int(np.floor(cy - radius))))
    x2 = max(x1 + 1, min(width, int(np.ceil(cx + radius))))
    y2 = max(y1 + 1, min(height, int(np.ceil(cy + radius))))
    return x1, y1, x2, y2


def polygon_in_region(polygon, region, width: int, height: int) -> bool:
    region = normalize_region(region)
    if not isinstance(region, dict):
        return polygon_in_roi(polygon, region, width, height)
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    area = cv2.contourArea(points)
    if area < 25:
        return False
    cx, cy, radius = circle_pixels(region, width, height)
    if np.linalg.norm(points.mean(axis=0) - [cx, cy]) > radius:
        return False
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.column_stack((cx + radius * np.cos(angles),
                              cy + radius * np.sin(angles))).astype(np.float32)
    overlap, _ = cv2.intersectConvexConvex(points, circle)
    return overlap / area >= 0.5


def region_near_edge(polygon, region, width: int, height: int) -> bool:
    points = np.asarray(polygon, dtype=np.float32)
    if isinstance(region, dict):
        cx, cy, radius = circle_pixels(region, width, height)
        return bool(np.any(np.linalg.norm(points - [cx, cy], axis=1) >= radius - 3))
    x1, y1, x2, y2 = region_bounds(region, width, height)
    return bool(np.any(points[:, 0] <= x1 + 2) or
                np.any(points[:, 0] >= x2 - 3) or
                np.any(points[:, 1] <= y1 + 2) or
                np.any(points[:, 1] >= y2 - 3))


def crop_region(image, region):
    """Crop to the region bounds and mask pixels outside a circle."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region_bounds(region, width, height)
    crop = image[y1:y2, x1:x2].copy()
    if isinstance(region, dict):
        cx, cy, radius = circle_pixels(region, width, height)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (round(cx - x1), round(cy - y1)), round(radius), 255, -1)
        crop[mask == 0] = 0
    return crop
