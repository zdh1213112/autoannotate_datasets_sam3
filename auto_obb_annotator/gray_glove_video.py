"""Repeatable, resumable frame extraction for the gray glove annotator."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import cv2

from .orbbec_undistort import camera_id_for_video
from .gray_glove_mcap import CAMERA_TOPICS, iter_mcap_frames, probe_mcap, validate_mcap_files


VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mcap"})


def raw_frame_path(folder: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("抽帧图片名称无效")
    return Path(folder) / "raw_frames" / Path(name).with_suffix(".png")


def cache_raw_frame(folder: Path, name: str, image) -> Path:
    """Cache only selected raw frames, losslessly and outside the image list."""
    destination = raw_frame_path(folder, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.png")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"无法保存原始帧缓存: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def frame_camera_ids(folder: Path) -> dict[str, str]:
    """Read source-camera IDs from batch manifests when available."""
    result = {}
    for manifest_path in Path(folder).glob("extract_*.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frames = manifest["frames"]
            transform = manifest["settings"].get("transform", {})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"抽帧来源记录无法读取: {manifest_path}") from exc
        for frame in frames:
            video = frame["video"]
            camera_id = frame.get("camera_id") or transform.get(video, {}).get("camera_id")
            if camera_id is None:
                try:
                    camera_id = camera_id_for_video(Path(video))
                except ValueError:
                    continue
            result[frame["name"]] = camera_id
    return result


def collect_videos(paths) -> list[Path]:
    """Expand folders recursively and return unique videos in a stable order."""
    videos = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            videos.update(item.resolve() for item in path.rglob("*")
                          if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos.add(path)
        else:
            raise ValueError(f"不是受支持的录像或录像目录: {path}")
    if not videos:
        raise ValueError("没有找到可抽帧的视频或 MCAP")
    return sorted(videos, key=lambda item: str(item))


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _frame_plan(video: Path, interval_mode: str, interval_value: float,
                start_seconds: float, end_seconds: float | None) -> tuple[float, int, list[int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"视频无法打开: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError(f"视频缺少有效帧率或帧数: {video}")
    first = max(0, math.ceil(start_seconds * fps))
    stop = min(frame_count, math.ceil(end_seconds * fps) if end_seconds is not None
               else frame_count)
    if interval_mode == "frames":
        indices = list(range(first, stop, int(interval_value)))
    else:
        indices = []
        last_index = -1
        limit = min(frame_count / fps, end_seconds if end_seconds is not None
                    else frame_count / fps)
        sample = 0
        while True:
            timestamp = start_seconds + sample * interval_value
            if timestamp >= limit - 1e-9:
                break
            index = math.ceil(timestamp * fps - 1e-9)
            if index > last_index and index < stop:
                indices.append(index)
                last_index = index
            sample += 1
    return fps, frame_count, indices


def extract_video_batch(videos, output: Path, interval_mode: str,
                        interval_value: float, start_seconds: float = 0.0,
                        end_seconds: float | None = None, progress=None,
                        should_continue=None, target_dir: Path | None = None,
                        frame_transform=None, transform_info: dict | None = None,
                        mcap_camera_ids=None
                        ) -> tuple[Path, int, int]:
    """Append reproducibly named frames to one label directory and resume partial runs.

    Returns (directory, total_selected_frames, frames_reused). Existing JPEGs are
    never overwritten. A manifest records source paths and frame indices.
    """
    videos = collect_videos(videos)
    mcap_files = [item for item in videos if item.suffix.lower() == ".mcap"]
    if mcap_files:
        validate_mcap_files(mcap_files)
        regular = [item for item in videos if item.suffix.lower() != ".mcap"]
        total = reused = 0
        folder = None
        if regular:
            folder, count, reused_here = extract_video_batch(
                regular, output, interval_mode, interval_value, start_seconds,
                end_seconds, progress, should_continue, target_dir,
                frame_transform, transform_info)
            total += count
            reused += reused_here
        for mcap in mcap_files:
            folder, count, reused_here = _extract_mcap_batch(
                mcap, output, interval_mode, interval_value, start_seconds,
                end_seconds, progress, should_continue, target_dir,
                frame_transform, transform_info, mcap_camera_ids)
            total += count
            reused += reused_here
        return folder, total, reused
    if interval_mode not in {"frames", "seconds"}:
        raise ValueError("抽帧方式须为按帧或按秒")
    if not math.isfinite(interval_value) or interval_value <= 0:
        raise ValueError("抽帧间隔必须大于 0")
    if interval_mode == "frames" and not float(interval_value).is_integer():
        raise ValueError("按帧抽取时，间隔必须是整数")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("开始时间不能小于 0")
    if end_seconds is not None and (not math.isfinite(end_seconds)
                                    or end_seconds <= start_seconds):
        raise ValueError("结束时间必须大于开始时间")
    if (frame_transform is None) != (transform_info is None):
        raise ValueError("去畸变处理与来源记录必须同时指定")
    output = Path(output).expanduser().resolve()
    target = (Path(target_dir).expanduser().resolve() if target_dir is not None
              else output / "video_frames" / "collection")
    if target != output / "video_frames" / "collection" and not target.is_relative_to(
            output / "video_frames"):
        raise ValueError("抽帧目录必须位于结果目录的 video_frames 中")
    settings = {"interval_mode": interval_mode, "interval_value": interval_value,
                "start_seconds": start_seconds, "end_seconds": end_seconds}
    if transform_info is not None:
        settings["transform"] = transform_info
    planned = []
    source_info = []
    for video in videos:
        stat = video.stat()
        fps, source_count, indices = _frame_plan(
            video, interval_mode, interval_value, start_seconds, end_seconds)
        source_settings = {key: value for key, value in settings.items()
                           if key != "transform"}
        if transform_info is not None:
            source_settings["transform"] = transform_info[str(video)]
        signature = json.dumps([str(video), stat.st_size, stat.st_mtime_ns, source_settings],
                               sort_keys=True)
        source_id = hashlib.sha256(signature.encode()).hexdigest()[:10]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem).strip("_")[:48] or "video"
        prefix = f"{stem}_{source_id}"
        source_info.append({"video": str(video), "fps": fps,
                            "source_frame_count": source_count,
                            "selected_count": len(indices), "prefix": prefix})
        planned.extend((video, fps, index, f"{prefix}__frame_{index:09d}.jpg")
                       for index in indices)
    if not planned:
        raise ValueError("所选时间范围内没有可抽取的帧")
    batch_key = hashlib.sha256(json.dumps(
        [source_info, settings], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / f"extract_{batch_key}.json"
    manifest = {"format_version": 1, "settings": settings, "sources": source_info,
                "frames": [{"name": name, "video": str(video),
                            "source_frame": index, "time_seconds": round(index / fps, 6)}
                           for video, fps, index, name in planned],
                "complete": False}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != manifest[key]
               for key in ("settings", "sources", "frames")):
            raise ValueError(f"抽帧记录与当前视频不一致，请检查: {manifest_path}")
    _write_json_atomic(manifest_path, manifest)
    total = len(planned)
    completed = 0
    reused = 0
    current_video = None
    capture = None
    try:
        for video, _fps, index, name in planned:
            if should_continue is not None and not should_continue():
                raise InterruptedError(f"已停止抽帧，已完成 {completed}/{total} 张；再次点击可继续")
            destination = target / name
            if destination.is_file() and destination.stat().st_size > 0:
                reused += 1
            else:
                if video != current_video:
                    if capture is not None:
                        capture.release()
                    capture = cv2.VideoCapture(str(video))
                    if not capture.isOpened():
                        raise ValueError(f"视频无法打开: {video}")
                    current_video = video
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"视频第 {index} 帧无法解码: {video}")
                if frame_transform is not None:
                    cache_raw_frame(target, name, image)
                    image = frame_transform(video, image)
                temporary = target / f".{name}.{uuid.uuid4().hex}.jpg"
                try:
                    if not cv2.imwrite(str(temporary), image,
                                       [cv2.IMWRITE_JPEG_QUALITY, 93]):
                        raise OSError(f"无法写入抽帧图片: {name}")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            completed += 1
            if progress:
                progress(completed, total)
    finally:
        if capture is not None:
            capture.release()
    manifest["complete"] = True
    _write_json_atomic(manifest_path, manifest)
    return target, total, reused


def _extract_mcap_batch(mcap, output, interval_mode, interval_value,
                        start_seconds, end_seconds, progress, should_continue,
                        target_dir, frame_transform, transform_info, mcap_camera_ids):
    if interval_mode not in {"frames", "seconds"} or not math.isfinite(interval_value) or interval_value <= 0:
        raise ValueError("抽帧方式或间隔无效")
    if interval_mode == "frames" and not float(interval_value).is_integer():
        raise ValueError("按帧抽取时，间隔必须是整数")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("开始时间不能小于 0")
    if end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start_seconds):
        raise ValueError("结束时间必须大于开始时间")
    if (frame_transform is None) != (transform_info is None):
        raise ValueError("去畸变处理与来源记录必须同时指定")
    mcap = Path(mcap).expanduser().resolve()
    streams = probe_mcap(mcap, mcap_camera_ids)
    output = Path(output).expanduser().resolve()
    target = (Path(target_dir).expanduser().resolve() if target_dir is not None
              else output / "video_frames" / "collection")
    if target != output / "video_frames" / "collection" and not target.is_relative_to(
            output / "video_frames"):
        raise ValueError("抽帧目录必须位于结果目录的 video_frames 中")
    stat = mcap.stat()
    settings = {"interval_mode": interval_mode, "interval_value": interval_value,
                "start_seconds": start_seconds, "end_seconds": end_seconds,
                "mcap_camera_ids": [stream.camera_id for stream in streams]}
    if transform_info is not None:
        settings["transform"] = {stream.key: transform_info[stream.key] for stream in streams}
    source_info = []
    for stream in streams:
        source_settings = {key: value for key, value in settings.items()
                           if key not in {"mcap_camera_ids", "transform"}}
        if transform_info is not None:
            source_settings["transform"] = transform_info[stream.key]
        signature = [str(mcap), stat.st_size, stat.st_mtime_ns, stream.topic,
                     source_settings]
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:10]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", mcap.stem).strip("_")[:40] or "mcap"
        source_info.append({"video": str(mcap), "topic": stream.topic,
                            "camera_id": stream.camera_id,
                            "source_message_count": stream.message_count,
                            "prefix": f"{stem}_{stream.camera_id}_{digest}"})
    batch_key = hashlib.sha256(json.dumps(
        [source_info, settings], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / f"extract_{batch_key}.json"
    manifest = {"format_version": 2, "settings": settings,
                "sources": source_info, "frames": [], "complete": False}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("settings") != settings or previous.get("sources") != source_info:
            raise ValueError(f"抽帧记录与当前 MCAP 不一致，请检查: {manifest_path}")
    _write_json_atomic(manifest_path, manifest)
    prefixes = {item["topic"]: item["prefix"] for item in source_info}
    range_indices = {stream.topic: 0 for stream in streams}
    next_times = {stream.topic: start_seconds for stream in streams}
    completed = reused = 0
    estimated = 0
    for stream in streams:
        duration = (stream.last_log_time_ns - stream.first_log_time_ns) / 1e9
        stop = min(duration + 1e-6, end_seconds) if end_seconds is not None else duration + 1e-6
        if stop <= start_seconds:
            continue
        if interval_mode == "seconds":
            estimated += math.ceil((stop - start_seconds) / interval_value)
        else:
            count = stream.message_count * (stop - start_seconds) / max(duration, 1e-6)
            estimated += math.ceil(count / interval_value)
    ended = set()
    try:
        for stream, index, seconds, image in iter_mcap_frames(
                mcap, streams, should_continue):
            if end_seconds is not None and seconds >= end_seconds:
                ended.add(stream.topic)
                if len(ended) == len(streams):
                    break
                continue
            if seconds + 1e-9 < start_seconds:
                continue
            topic = stream.topic
            if interval_mode == "frames":
                selected = range_indices[topic] % int(interval_value) == 0
            else:
                selected = seconds + 1e-9 >= next_times[topic]
            range_indices[topic] += 1
            if not selected:
                continue
            if interval_mode == "seconds":
                while next_times[topic] <= seconds + 1e-9:
                    next_times[topic] += interval_value
            name = f"{prefixes[topic]}__frame_{index:09d}.jpg"
            destination = target / name
            if destination.is_file() and destination.stat().st_size > 0:
                reused += 1
            else:
                if hasattr(image, "to_ndarray"):
                    image = image.to_ndarray(format="bgr24")
                if frame_transform is not None:
                    cache_raw_frame(target, name, image)
                    image = frame_transform((mcap, topic), image)
                temporary = target / f".{name}.{uuid.uuid4().hex}.jpg"
                try:
                    if not cv2.imwrite(str(temporary), image,
                                       [cv2.IMWRITE_JPEG_QUALITY, 93]):
                        raise OSError(f"无法写入抽帧图片: {name}")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            manifest["frames"].append({"name": name, "video": str(mcap),
                                       "topic": topic, "camera_id": stream.camera_id,
                                       "source_frame": index,
                                       "time_seconds": round(seconds, 6)})
            completed += 1
            if completed % 20 == 0:
                _write_json_atomic(manifest_path, manifest)
            if progress:
                progress(completed, max(completed, estimated))
    except Exception:
        _write_json_atomic(manifest_path, manifest)
        raise
    if not completed:
        raise ValueError(f"所选时间范围内没有可抽取的 MCAP 图像: {mcap}")
    manifest["complete"] = True
    _write_json_atomic(manifest_path, manifest)
    if progress:
        progress(completed, completed)
    return target, completed, reused


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
