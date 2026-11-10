"""Small, representative visual checks drawn from the exported YOLO labels."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
from urllib.parse import quote

import cv2
from PIL import Image, ImageDraw, ImageFont


PREVIEW_LIMIT = 10
COLORS = ((0, 200, 0), (0, 180, 255), (255, 170, 0),
          (220, 90, 210), (255, 100, 100), (100, 210, 230))


def choose_preview_records(records, val_names, limit=PREVIEW_LIMIT):
    """Cover splits, classes, cameras and empty images before filling the sample."""
    ordered = sorted(records, key=lambda record: (
        hashlib.sha256(record["name"].encode()).hexdigest(), record["name"]))
    limit = min(limit, len(ordered))
    chosen, covered = [], set()

    def features(record):
        objects = record.get("objects", [])
        values = {("split", "val" if record["name"] in val_names else "train"),
                  ("count", "empty" if not objects else "single" if len(objects) == 1 else "multiple")}
        values.update(("class", obj["class_id"]) for obj in objects)
        if record.get("camera_id"):
            values.add(("camera", record["camera_id"]))
        return values

    remaining = [(record, features(record)) for record in ordered]
    while remaining and len(chosen) < limit:
        index = max(range(len(remaining)), key=lambda index: len(remaining[index][1] - covered))
        record, values = remaining[index]
        if not values - covered:
            break
        remaining.pop(index)
        chosen.append(record)
        covered.update(values)
    # Evenly spread the remaining slots over filenames instead of taking ten
    # adjacent frames from the beginning of a recording.
    remaining.sort(key=lambda pair: pair[0]["name"])
    slots = limit - len(chosen)
    if slots:
        for index in range(slots):
            chosen.append(remaining[(2 * index + 1) * len(remaining) // (2 * slots)][0])
    return sorted(chosen, key=lambda record: (record["name"] in val_names, record["name"]))


@lru_cache(maxsize=4)
def _font(size):
    for filename in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_preview(image_path, label_path, classes, box_format, subset):
    pixels = cv2.imread(str(image_path))
    if pixels is None:
        raise ValueError(f"无法读取可视化样例图片: {image_path}")
    image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    image.thumbnail((1600, 1200))
    width, height = image.size
    header = 54
    canvas = Image.new("RGB", (width, height + header), (28, 32, 38))
    canvas.paste(image, (0, header))
    draw, font = ImageDraw.Draw(canvas), _font(18)
    rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    title = f"{subset} | {image_path.name}"
    while len(title) > 4 and draw.textlength(title, font=font) > width - 12:
        title = title[:-4] + "..."
    draw.text((6, 2), title, font=font, fill="white")
    draw.text((6, 27), f"{box_format.upper()} | {len(rows)} objects" if rows else
              f"{box_format.upper()} | NO OBJECTS", font=font, fill="white")
    for row in rows:
        class_id = int(row[0])
        values = [float(value) for value in row[1:]]
        if box_format == "hbb" and len(values) == 4:
            cx, cy, w, h = values
            points = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                      (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
        elif box_format == "obb" and len(values) == 8:
            points = list(zip(values[::2], values[1::2]))
        else:
            raise ValueError(f"可视化样例的 YOLO 标签格式无效: {label_path}")
        points = [(max(0, min(width - 1, round(x * width))),
                   header + max(0, min(height - 1, round(y * height)))) for x, y in points]
        color = COLORS[class_id % len(COLORS)]
        draw.line([*points, points[0]], fill=color, width=3)
        text = f"{class_id}: {classes[class_id]['name']}"
        box = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        x = max(0, min(min(point[0] for point in points), width - text_width - 8))
        y = max(header, min(point[1] for point in points) - text_height - 8)
        draw.rectangle((x, y, x + text_width + 8, y + text_height + 6), fill=(28, 32, 38))
        draw.text((x + 4 - box[0], y + 3 - box[1]), text, font=font, fill=color)
    return canvas


def write_dataset_previews(target, records, classes, box_format, val_names):
    """Write up to ten annotated images, a contact sheet and a sample index."""
    samples = choose_preview_records(records, val_names)
    folder = target / "visualizations"
    folder.mkdir()
    thumbs, lines = [], []
    for record in samples:
        name = record["name"]
        subset = "val" if name in val_names else "train"
        image = target / "images" / subset / name
        label = target / "labels" / subset / f"{Path(name).stem}.txt"
        filename = f"{subset}__{Path(name).stem}.jpg"
        preview = _draw_preview(image, label, classes, box_format, subset)
        preview.save(folder / filename, quality=92)
        preview.thumbnail((480, 340))
        thumbs.append(preview)
        lines.append(
            f"| [{name}]({quote(filename)}) | {subset} | "
            f"[原图]({quote('../images/' + subset + '/' + name)}) | "
            f"[标签]({quote('../labels/' + subset + '/' + Path(name).stem + '.txt')}) |")
    columns = min(2, len(thumbs))
    rows = math.ceil(len(thumbs) / columns)
    overview = Image.new("RGB", (columns * 492 + 12, rows * 352 + 12), (40, 44, 50))
    for index, thumb in enumerate(thumbs):
        x = 12 + (index % columns) * 492 + (480 - thumb.width) // 2
        y = 12 + (index // columns) * 352 + (340 - thumb.height) // 2
        overview.paste(thumb, (x, y))
    overview.save(folder / "overview.jpg", quality=92)
    class_lines = "\n".join(f"- `{index}`: {spec['name']}" for index, spec in enumerate(classes))
    (folder / "README.md").write_text(
        f"# 数据集可视化预览\n\n从本次导出的 {len(records)} 张图片中选择 {len(samples)} 张，"
        "优先覆盖训练/验证集、类别、相机视角及无目标样本，其余名额分散选取。\n\n"
        f"框根据本数据集实际的 `labels/` 文件绘制，格式为 **{box_format.upper()}**。"
        "预览仅供抽查；训练使用 `images/`、`labels/` 和 `dataset.yaml`。\n\n"
        "先打开 [overview.jpg](overview.jpg) 查看总览，再打开单张样例查看细节。\n\n"
        f"## 类别\n\n{class_lines}\n\n"
        "## 样例来源\n\n| 可视化样例 | 划分 | 原图 | 标签 |\n| --- | --- | --- | --- |\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    return len(samples)
