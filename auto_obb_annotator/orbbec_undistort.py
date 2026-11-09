"""KB fisheye undistortion for sampled Orbbec EGO4 camera frames."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    stem = Path(video).stem.lower()
    for suffix, camera_id in CAMERA_SUFFIXES:
        if stem.endswith(suffix):
            return camera_id
    raise ValueError(
        f"无法判断视频对应的四目相机: {Path(video).name}；"
        "文件名须以 camera_left、camera_right、camera_right_side 或 camera_left_side 结尾")


def calibration_for_video(video: Path, override: Path | None = None) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"标定文件不存在: {path}")
        return path
    video = Path(video).expanduser().resolve()
    matches = sorted(video.parent.glob("*_calibration_camera.yaml"))
    session_prefix = video.stem.split("_camera_")[0]
    exact = [path for path in matches if path.name == f"{session_prefix}_calibration_camera.yaml"]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"未在视频目录找到唯一的 *_calibration_camera.yaml: {video.parent}；"
        "请手动选择对应标定文件")


class OrbbecUndistorter:
    """Build one OpenCV fisheye map per camera and reuse it for selected frames."""

    def __init__(self, videos: list[Path], calibration_override: Path | None = None,
                 balance: float = 0.0):
        if not 0 <= balance <= 1:
            raise ValueError("保留视野比例须在 0～1 之间")
        self.balance = float(balance)
        self._by_video = {}
        self._maps = {}
        self.metadata = {}
        parsed = {}
        for raw_video in videos:
            video = Path(raw_video).expanduser().resolve()
            camera_id = camera_id_for_video(video)
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
                parsed[calibration_path] = (cameras, hashlib.sha256(payload).hexdigest(),
                                            str(data.get("calibration_info", {}).get("serial_number", "")))
            cameras, digest, serial = parsed[calibration_path]
            if (video.stem.startswith("EGO-") and serial
                    and f"_{serial}_" not in video.stem):
                raise ValueError(
                    f"视频 {video.name} 与标定文件序列号 {serial} 不匹配")
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
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise ValueError(f"视频无法打开: {video}")
                actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            finally:
                capture.release()
            if actual_size != size:
                raise ValueError(
                    f"{video.name} 的实际画面 {actual_size[0]}×{actual_size[1]} "
                    f"与 {camera_id} 标定 {size[0]}×{size[1]} 不一致")
            rotation = np.eye(3, dtype=np.float64)
            new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix, coefficients, size, rotation, balance=self.balance, new_size=size)
            self._by_video[video] = (camera_id, size, matrix, coefficients,
                                     new_matrix, digest)
            self.metadata[str(video)] = {
                "camera_id": camera_id, "calibration_path": str(calibration_path),
                "calibration_sha256": digest, "calibration_serial": serial,
                "image_size": list(size), "balance": self.balance,
                "projection": "independent_kb_fisheye_undistortion",
                "new_camera_matrix": new_matrix.tolist(),
            }

    def __call__(self, video: Path, frame: np.ndarray) -> np.ndarray:
        video = Path(video).expanduser().resolve()
        camera_id, size, matrix, coefficients, new_matrix, digest = self._by_video[video]
        if (frame.shape[1], frame.shape[0]) != size:
            raise ValueError(
                f"{video.name} 的实际画面 {frame.shape[1]}×{frame.shape[0]} "
                f"与 {camera_id} 标定 {size[0]}×{size[1]} 不一致")
        key = (digest, camera_id, size, self.balance)
        if key not in self._maps:
            rotation = np.eye(3, dtype=np.float64)
            self._maps[key] = cv2.fisheye.initUndistortRectifyMap(
                matrix, coefficients, rotation, new_matrix, size, cv2.CV_32FC1)
        map_x, map_y = self._maps[key]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
