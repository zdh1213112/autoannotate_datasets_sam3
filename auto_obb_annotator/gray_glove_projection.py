"""Segment raw sampled frames, then project instance masks into review images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .generic_region import crop_region, region_bounds
from .gray_glove_mcap import iter_mcap_frames, probe_mcap
from .gray_glove_video import cache_raw_frame, raw_frame_path
from .orbbec_undistort import OrbbecUndistorter


PROJECTION_WORKFLOW = "raw_sam3_masks_to_undistorted_v1"


def frame_projection_sources(folder: Path) -> dict[str, dict]:
    """Index transformed frames without needing the original recordings online."""
    result = {}
    for path in sorted(Path(folder).glob("extract_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        settings = manifest["settings"]
        transforms = settings.get("transform", {})
        if not transforms:
            continue
        for frame in manifest["frames"]:
            key = frame["video"] + ("::" + frame["topic"] if frame.get("topic") else "")
            metadata = transforms.get(key)
            if not metadata or metadata.get("projection") != "independent_kb_fisheye_undistortion":
                raise ValueError(f"抽帧缺少可同步标签的去畸变参数: {path} {frame['name']}")
            entry = {**frame, "source_key": key, "transform": metadata, "settings": settings}
            if frame["name"] in result and result[frame["name"]] != entry:
                raise ValueError(f"图片对应多个不同去畸变来源: {frame['name']}")
            result[frame["name"]] = entry
    return result


class RawFrameReader:
    """Read cached raw PNGs, or recover old sampled frames from MP4/MCAP once."""

    def __init__(self, folder: Path, should_continue=None):
        self.folder = Path(folder)
        self.entries = frame_projection_sources(folder)
        self.should_continue = should_continue
        self._captures = {}
        self._mcap_readers = {}
        self._rectifiers = {}
        self._verified_sources = set()

    def sort_key(self, path: Path):
        entry = self.entries.get(path.name)
        return ((entry["source_key"], entry["source_frame"]) if entry else (str(path), -1))

    def _verify_source(self, entry):
        key = entry["source_key"]
        if key in self._verified_sources:
            return
        video = Path(entry["video"])
        if not video.is_file():
            raise FileNotFoundError(f"缺少原图缓存和原始录像，无法同步标签: {video}")
        stat = video.stat()
        settings = {key: value for key, value in entry["settings"].items()
                    if key not in {"transform", "mcap_camera_ids"}}
        settings["transform"] = entry["transform"]
        signature = [str(video), stat.st_size, stat.st_mtime_ns]
        if entry.get("topic"):
            signature.append(entry["topic"])
        signature.append(settings)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:10]
        if f"_{digest}__frame_" not in entry["name"]:
            raise ValueError(f"原始录像已改变，与去畸变抽帧来源不一致: {video}")
        self._verified_sources.add(key)

    def _read_mcap(self, entry):
        key, wanted = entry["source_key"], entry["source_frame"]
        previous, iterator = self._mcap_readers.get(key, (-1, None))
        if iterator is None or wanted <= previous:
            if iterator is not None:
                iterator.close()
            streams = probe_mcap(Path(entry["video"]), [entry["transform"]["camera_id"]])
            iterator = iter_mcap_frames(Path(entry["video"]), streams, self.should_continue)
        self._mcap_readers[key] = (wanted, iterator)
        for stream, index, seconds, image in iterator:
            if stream.topic != entry["topic"] or index < wanted:
                continue
            if index > wanted or abs(seconds - entry["time_seconds"]) > 1e-5:
                break
            return image.to_ndarray(format="bgr24") if hasattr(image, "to_ndarray") else image
        raise ValueError(f"原始 MCAP 中未找到对应帧: {key} 第 {wanted} 帧")

    def load(self, name: str):
        entry = self.entries[name]
        key = entry["source_key"]
        projection_key = (key, json.dumps(entry["transform"], sort_keys=True))
        if projection_key not in self._rectifiers:
            self._rectifiers[projection_key] = OrbbecUndistorter.from_metadata(key, entry["transform"])
        rectifier = self._rectifiers[projection_key]
        cached = raw_frame_path(self.folder, name)
        if cached.is_file():
            image = cv2.imread(str(cached))
        else:
            self._verify_source(entry)
            if entry.get("topic"):
                image = self._read_mcap(entry)
            else:
                video = entry["video"]
                if video not in self._captures:
                    self._captures[video] = cv2.VideoCapture(video)
                capture = self._captures[video]
                capture.set(cv2.CAP_PROP_POS_FRAMES, entry["source_frame"])
                ok, image = capture.read()
                if not ok:
                    image = None
            if image is not None:
                rectifier.maps_for(key, image.shape)
                cache_raw_frame(self.folder, name, image)
        if image is None:
            raise ValueError(f"原始帧无法解码: {name}")
        rectifier.maps_for(key, image.shape)
        return image, rectifier, key

    def close(self):
        for capture in self._captures.values():
            capture.release()
        for _, iterator in self._mcap_readers.values():
            iterator.close()
        self._captures.clear()
        self._mcap_readers.clear()


class ProjectedGloveSegmenter:
    """Adapt raw SAM3 detections to the ROI coordinates of an undistorted frame."""

    def __init__(self, segmenter, raw_image, rectifier, source_key, roi, output_shape):
        self.segmenter = segmenter
        self.raw_image = raw_image
        self.rectifier = rectifier
        self.source_key = source_key
        self.roi = roi
        self.last_side_detections = []
        if raw_image.shape[:2] != output_shape[:2]:
            raise ValueError("去畸变图片尺寸与抽帧标定不一致，无法同步标签")
        self.bounds = region_bounds(roi, output_shape[1], output_shape[0])

    def _project(self, detection, require_mask=True):
        height, width = self.raw_image.shape[:2]
        x1, y1, x2, y2 = detection["box"]
        xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xb, yb = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if xb <= xa or yb <= ya:
            return None
        mask = detection.get("mask")
        if mask is None:
            if require_mask:
                raise ValueError("SAM3 未返回原图分割掩码，无法精确同步去畸变标签")
            mask = np.ones((height, width), dtype=np.uint8)
        else:
            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError("SAM3 分割掩码尺寸无效")
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        restricted = np.zeros((height, width), dtype=np.uint8)
        restricted[ya:yb, xa:xb] = mask[ya:yb, xa:xb]
        projected = self.rectifier.remap_mask(self.source_key, restricted)
        cropped = crop_region(projected, self.roi)
        points = cv2.findNonZero(cropped)
        if points is None:
            return None  # Outside the retained view or the user-selected ROI.
        x, y, w, h = cv2.boundingRect(points)
        return {"class_id": detection["class_id"], "score": detection["score"],
                "box": [x, y, x + w, y + h], "mask": cropped, "pad_mask": False}

    def detect_many(self, _undistorted_crop, prompts):
        raw_detections = self.segmenter.detect_many(self.raw_image, prompts)
        self.last_side_detections = [projected for detection in
                                     getattr(self.segmenter, "last_side_detections", [])
                                     if (projected := self._project(detection, require_mask=False)) is not None]
        return [projected for detection in raw_detections
                if (projected := self._project(detection)) is not None]
