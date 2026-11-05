"""Repeatable, resumable frame extraction for the gray glove annotator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2


def video_frame_directory(video: Path, output: Path, interval_seconds: float) -> Path:
    """Keep source videos and extraction settings in separate frame folders."""
    video = video.expanduser().resolve()
    stat = video.stat()
    signature = f"{video}|{stat.st_size}|{stat.st_mtime_ns}|{interval_seconds:.3f}"
    digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
    return output.expanduser().resolve() / "video_frames" / f"{video.stem[:50]}_{digest}"


def extract_video_frames(video: Path, output: Path, interval_seconds: float,
                         progress=None, should_continue=None) -> tuple[Path, int, bool]:
    """Extract evenly spaced JPEGs, publishing only a complete frame set.

    Returns ``(frame_directory, frame_count, reused)``. A repeat import with
    unchanged source and interval reuses the same frames and label project.
    """
    video = Path(video).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video}")
    if video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError("请选择 MP4、MOV、AVI 或 MKV 视频")
    if not 0.1 <= interval_seconds <= 60:
        raise ValueError("抽帧间隔须在 0.1～60 秒之间")
    target = video_frame_directory(video, output, interval_seconds)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = manifest["frames"]
            if (manifest["video"] == str(video) and
                    manifest["interval_seconds"] == interval_seconds and
                    names and all((target / name).is_file() for name in names)):
                if progress:
                    progress(len(names), len(names))
                return target, len(names), True
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise ValueError(f"已有不完整的抽帧目录：{target}；请先检查或移走该目录")
    if target.exists():
        raise ValueError(f"已有不完整的抽帧目录：{target}；请先检查或移走该目录")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"视频无法打开: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("视频缺少有效的帧率或总帧数")
    step = max(1, round(interval_seconds * fps))
    indices = list(range(0, frame_count, step))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    names = []
    try:
        for position, frame_index in enumerate(indices, start=1):
            if should_continue is not None and not should_continue():
                raise InterruptedError("已停止抽帧；未保存不完整图片")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"视频第 {frame_index} 帧无法解码")
            name = f"frame_{frame_index:09d}.jpg"
            if not cv2.imwrite(str(temporary / name), image,
                               [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(f"无法写入抽帧图片: {name}")
            names.append(name)
            if progress:
                progress(position, len(indices))
        (temporary / "manifest.json").write_text(json.dumps({
            "video": str(video), "interval_seconds": interval_seconds,
            "fps": fps, "source_frame_count": frame_count, "frames": names,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target, len(names), False
    finally:
        capture.release()
        if temporary.exists():
            shutil.rmtree(temporary)
