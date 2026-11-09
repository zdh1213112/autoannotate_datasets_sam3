"""Stream Orbbec Foxglove camera frames from MCAP without converting videos."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


CAMERA_TOPICS = {
    "cam_0": "/camera/left",
    "cam_1": "/camera/right",
    "cam_2": "/camera/right_side",
    "cam_3": "/camera/left_side",
}


def _dependencies():
    try:
        import av
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError(
            "读取 MCAP 需要 mcap、mcap-protobuf-support 和 av；"
            "请在当前 Python 环境安装 requirements.txt") from exc
    return av, make_reader, DecoderFactory


@dataclass(frozen=True)
class McapStream:
    path: Path
    topic: str
    camera_id: str
    message_count: int
    first_log_time_ns: int
    last_log_time_ns: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.topic}"


def probe_mcap(path: Path, camera_ids=None) -> list[McapStream]:
    """Find the four known EGO4 image topics and their time origins."""
    _av, make_reader, _factory = _dependencies()
    path = Path(path).expanduser().resolve()
    wanted = set(CAMERA_TOPICS if camera_ids is None else camera_ids)
    unknown = wanted - CAMERA_TOPICS.keys()
    if unknown:
        raise ValueError(f"未知 MCAP 相机: {', '.join(sorted(unknown))}")
    topics = {topic: camera for camera, topic in CAMERA_TOPICS.items() if camera in wanted}
    if not topics:
        raise ValueError("至少选择一路 MCAP 相机")
    first = {}
    last = {}
    counts = {topic: 0 for topic in topics}
    with path.open("rb") as source:
        reader = make_reader(source)
        for schema, channel, message in reader.iter_messages(
                topics=list(topics), log_time_order=False):
            if (schema is None or schema.name not in {
                    "foxglove.CompressedImage", "foxglove.CompressedVideo"}
                    or channel.message_encoding != "protobuf"):
                continue
            topic = channel.topic
            first.setdefault(topic, int(message.log_time))
            last[topic] = int(message.log_time)
            counts[topic] += 1
    missing = set(topics) - set(first)
    if missing:
        raise ValueError(f"MCAP 缺少相机图像流: {', '.join(sorted(missing))}；{path}")
    return [McapStream(path, topic, camera, counts[topic], first[topic], last[topic])
            for camera, topic in CAMERA_TOPICS.items() if camera in wanted]


def iter_mcap_frames(path: Path, streams: list[McapStream], should_continue=None):
    """Yield frames in file order; H264 pixels are materialized only if selected."""
    av, make_reader, DecoderFactory = _dependencies()
    by_topic = {stream.topic: stream for stream in streams}
    decoders = {}
    source_indices = {topic: 0 for topic in by_topic}
    first_ns = {stream.topic: stream.first_log_time_ns for stream in streams}
    with Path(path).open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()])
        for schema, channel, message, image in reader.iter_decoded_messages(
                topics=list(by_topic), log_time_order=False):
            if should_continue is not None and not should_continue():
                raise InterruptedError("已停止抽帧；再次点击可继续")
            topic = channel.topic
            if schema is None or schema.name not in {
                    "foxglove.CompressedImage", "foxglove.CompressedVideo"}:
                continue
            index = source_indices[topic]
            source_indices[topic] += 1
            media_format = str(image.format).strip().lower().replace(".", "")
            data = bytes(image.data)
            if media_format in {"h264", "avc", "avc1"}:
                decoder = decoders.get(topic)
                if decoder is None:
                    decoder = av.CodecContext.create("h264", "r")
                    decoders[topic] = decoder
                packet = av.Packet(data)
                packet.pts = int(message.log_time)
                packet.dts = int(message.log_time)
                packet.time_base = Fraction(1, 1_000_000_000)
                try:
                    frames = decoder.decode(packet)
                except av.error.InvalidDataError:
                    # A segment may begin before the next H264 keyframe.
                    continue
                for frame in frames:
                    timestamp = int(frame.pts) if frame.pts is not None else int(message.log_time)
                    yield (by_topic[topic], index,
                           (timestamp - first_ns[topic]) / 1_000_000_000,
                           frame)
            elif media_format in {"jpeg", "jpg", "png"}:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"MCAP 图像无法解码: {path} {topic} 第 {index} 帧")
                yield (by_topic[topic], index,
                       (int(message.log_time) - first_ns[topic]) / 1_000_000_000, frame)
            else:
                raise ValueError(f"MCAP 图像格式暂不支持: {topic} {image.format!r}")
