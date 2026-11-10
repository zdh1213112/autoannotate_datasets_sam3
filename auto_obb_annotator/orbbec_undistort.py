"""KB fisheye undistortion for sampled Orbbec EGO4 camera frames."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import cv2
import numpy as np
import yaml


CAMERA_SUFFIXES = (
    ("_camera_left_side", "cam_3"),
    ("_camera_right_side", "cam_2"),
    ("_camera_left", "cam_0"),
    ("_camera_right", "cam_1"),
)


def camera_id_for_video(video: Path) -> str:
    stem = re.sub(r"_part\d+$", "", Path(video).stem.lower())
    for suffix, camera_id in CAMERA_SUFFIXES:
        if stem.endswith(suffix):
            return camera_id
    raise ValueError(
        f"无法判断视频对应的四目相机: {Path(video).name}；"
        "文件名须以 camera_left、camera_right、camera_right_side 或 camera_left_side "
        "结尾，可附加 _part0001 等分段编号")


def calibration_for_video(video: Path, override: Path | None = None) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"标定文件不存在: {path}")
        return path
    video = Path(video).expanduser().resolve()
    # MCAP parts live one directory below the session calibration.
    for folder in (video.parent, video.parent.parent):
        matches = sorted(set(folder.glob("*_calibration_camera.yaml")) |
                         set(folder.glob("*_params_cam_*.yaml")))
        session_prefix = video.stem.split("_camera_")[0].split("_part")[0]
        exact = [path for path in matches if path.name.startswith(session_prefix + "_")]
        if len(exact) == 1:
            return exact[0]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"未在录像目录找到唯一的相机标定 YAML: {video.parent}；"
        "请手动选择对应标定文件")


class OrbbecUndistorter:
    """Build one OpenCV fisheye map per camera and reuse it for selected frames."""

    def __init__(self, videos: list[Path], calibration_override: Path | None = None,
                 balance: float = 0.0, mcap_camera_ids=None):
        if not 0 <= balance <= 1:
            raise ValueError("保留视野比例须在 0～1 之间")
        self.balance = float(balance)
        self._by_video = {}
        self._maps = {}
        self.metadata = {}
        parsed = {}
        for raw_video in videos:
            video = Path(raw_video).expanduser().resolve()
            if video.suffix.lower() == ".mcap":
                from .gray_glove_mcap import probe_mcap
                sources = [(stream.key, stream.camera_id, None)
                           for stream in probe_mcap(video, mcap_camera_ids)]
            else:
                camera_id = camera_id_for_video(video)
                capture = cv2.VideoCapture(str(video))
                try:
                    if not capture.isOpened():
                        raise ValueError(f"视频无法打开: {video}")
                    actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                finally:
                    capture.release()
                sources = [(str(video), camera_id, actual_size)]
            calibration_path = calibration_for_video(video, calibration_override)
            if calibration_path not in parsed:
                payload = calibration_path.read_bytes()
                try:
                    data = yaml.safe_load(payload)
                except yaml.YAMLError as exc:
                    raise ValueError(f"相机标定 YAML 无法解析: {calibration_path}") from exc
                if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
                    raise ValueError(f"相机标定 YAML 缺少 cameras: {calibration_path}")
                cameras = {str(item.get("id")): item for item in data.get("cameras", [])}
                serial = str(data.get("calibration_info", {}).get("serial_number", ""))
                if not serial:
                    match = re.match(r"^EGO-[^_]+_([^_]+)_", calibration_path.name)
                    serial = match.group(1) if match else ""
                parsed[calibration_path] = (cameras, hashlib.sha256(payload).hexdigest(),
                                            serial)
            cameras, digest, serial = parsed[calibration_path]
            if (video.stem.startswith("EGO-") and serial
                    and f"_{serial}_" not in video.stem):
                raise ValueError(
                    f"视频 {video.name} 与标定文件序列号 {serial} 不匹配")
            for source_key, camera_id, actual_size in sources:
                if camera_id not in cameras:
                    raise ValueError(f"标定文件缺少 {camera_id}: {calibration_path}")
                camera = cameras[camera_id]
                if str(camera.get("distortion_model", "")).upper() != "KB":
                    raise ValueError(f"{camera_id} 的畸变模型不是 KB: {calibration_path}")
                size = (int(camera["image_width"]), int(camera["image_height"]))
                if min(size) <= 0:
                    raise ValueError(f"{camera_id} 的标定分辨率无效")
                intr = camera["intrinsics"]
                dist = camera["distortion"]
                matrix = np.array([[intr["fx"], 0, intr["cx"]],
                                   [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)
                coefficients = np.array([dist[f"k{index}"] for index in range(1, 5)],
                                        dtype=np.float64)
                if (not np.isfinite(matrix).all() or not np.isfinite(coefficients).all()
                        or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
                    raise ValueError(f"{camera_id} 的标定参数无效")
                if actual_size is not None and actual_size != size:
                    raise ValueError(
                        f"{video.name} 的实际画面 {actual_size[0]}×{actual_size[1]} "
                        f"与 {camera_id} 标定 {size[0]}×{size[1]} 不一致")
                rotation = np.eye(3, dtype=np.float64)
                new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    matrix, coefficients, size, rotation, balance=self.balance, new_size=size)
                self._by_video[source_key] = (camera_id, size, matrix, coefficients,
                                              new_matrix, digest)
                self.metadata[source_key] = {
                    "camera_id": camera_id, "calibration_path": str(calibration_path),
                    "calibration_sha256": digest, "calibration_serial": serial,
                    "image_size": list(size), "balance": self.balance,
                    "projection": "independent_kb_fisheye_undistortion",
                    "new_camera_matrix": new_matrix.tolist(),
                }

    @classmethod
    def from_metadata(cls, source_key: str, metadata: dict):
        """Reuse the exact saved projection, including its new camera matrix."""
        path = Path(metadata["calibration_path"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["calibration_sha256"]:
            raise ValueError(f"标定文件已改变，无法同步原图标签: {path}")
        data = yaml.safe_load(payload)
        camera_id = metadata["camera_id"]
        camera = next((item for item in data["cameras"] if item["id"] == camera_id), None)
        if camera is None or str(camera.get("distortion_model", "")).upper() != "KB":
            raise ValueError(f"标定文件缺少有效的 {camera_id} KB 参数: {path}")
        size = (int(camera["image_width"]), int(camera["image_height"]))
        if list(size) != metadata["image_size"]:
            raise ValueError("去畸变记录与标定分辨率不一致")
        intr, dist = camera["intrinsics"], camera["distortion"]
        matrix = np.array([[intr["fx"], 0, intr["cx"]],
                           [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)
        coefficients = np.array([dist[f"k{index}"] for index in range(1, 5)], dtype=np.float64)
        new_matrix = np.asarray(metadata["new_camera_matrix"], dtype=np.float64)
        if (new_matrix.shape != (3, 3) or not np.isfinite(new_matrix).all()
                or not np.isfinite(matrix).all() or not np.isfinite(coefficients).all()
                or min(matrix[0, 0], matrix[1, 1], new_matrix[0, 0], new_matrix[1, 1]) <= 0):
            raise ValueError("去畸变记录中的相机参数无效")
        instance = cls.__new__(cls)
        instance.balance = float(metadata["balance"])
        instance._maps = {}
        instance._by_video = {source_key: (camera_id, size, matrix, coefficients,
                                          new_matrix, metadata["calibration_sha256"])}
        instance.metadata = {source_key: dict(metadata)}
        return instance

    def maps_for(self, video, shape):
        """Return the shared output-to-input maps for images and masks."""
        source_key = (str(Path(video[0]).expanduser().resolve()) + "::" + video[1]
                      if isinstance(video, tuple) else str(video))
        if source_key not in self._by_video:
            source_key = str(Path(video).expanduser().resolve())
        camera_id, size, matrix, coefficients, new_matrix, digest = self._by_video[source_key]
        if (shape[1], shape[0]) != size:
            raise ValueError(
                f"{source_key} 的实际画面 {shape[1]}×{shape[0]} "
                f"与 {camera_id} 标定 {size[0]}×{size[1]} 不一致")
        key = (digest, camera_id, size, self.balance)
        if key not in self._maps:
            rotation = np.eye(3, dtype=np.float64)
            self._maps[key] = cv2.fisheye.initUndistortRectifyMap(
                matrix, coefficients, rotation, new_matrix, size, cv2.CV_32FC1)
        return self._maps[key]

    def __call__(self, video: Path, frame: np.ndarray) -> np.ndarray:
        map_x, map_y = self.maps_for(video, frame.shape)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def remap_mask(self, video, mask: np.ndarray) -> np.ndarray:
        """Use nearest-neighbor sampling to preserve a binary instance mask."""
        if mask.ndim != 2:
            raise ValueError("分割掩码必须为二维")
        map_x, map_y = self.maps_for(video, mask.shape)
        return cv2.remap((mask > 0).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
