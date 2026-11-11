"""Read and edit YOLO datasets without requiring a SAM3 review project."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import yaml

from .barcode_core import append_record
from .generic_core import sync_candidates, yolo_row, _write_yolo_export

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
CLASSES = [{'name': 'left', 'prompt': 'left glove'},
           {'name': 'right', 'prompt': 'right glove'}]


def dataset_root(folder):
    """Accept a root, images/labels folder, or a split below either folder."""
    folder = Path(folder).expanduser().resolve()
    candidates = []
    for candidate in (folder, *folder.parents):
        if (candidate / 'images').is_dir() or any(
                (candidate / split / 'images').is_dir() for split in ('train', 'val', 'valid', 'test')):
            candidates.append(candidate)
    for candidate in candidates:
        if any((candidate / name).is_file() for name in
               ('classes.txt', 'dataset.yaml', 'data.yaml', 'dataset.yml', 'data.yml')):
            return candidate
    if candidates:
        candidate = candidates[0]
        if candidate.name in {'train', 'val', 'valid', 'test'} and candidate.parent in candidates:
            return candidate.parent
        return candidate
    raise ValueError('请选择含 images/ 的 YOLO 数据集目录，或包含 project.json 的原复核项目。')


def _digest(data):
    return None if data is None else hashlib.sha256(data).hexdigest()


def _read(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.label_', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class YoloGloveDataset:
    def __init__(self, root):
        self.root = root
        self.records = {}
        self.labels = {}
        self.expected = {}
        self.formats = {}
        self.default_format = 'hbb'
        self.classes_assumed = False
        self.to_internal = {0: 0, 1: 1}
        self.to_disk = {0: 0, 1: 1}
        self.metadata = root / '.gray_glove_editor'
        self.metadata_files = []

    def inside(self, path):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError(f'标签或备份路径指向数据集外部，未写入：{path}')
        return path

    def read_classes(self):
        definitions = []
        classes_file = self.root / 'classes.txt'
        if classes_file.is_file():
            definitions.append([line.strip() for line in classes_file.read_text(encoding='utf-8-sig').splitlines()
                                if line.strip()])
            self.metadata_files.append(classes_file)
        for filename in ('dataset.yaml', 'data.yaml', 'dataset.yml', 'data.yml'):
            path = self.root / filename
            if not path.is_file():
                continue
            self.metadata_files.append(path)
            config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            names = config.get('names') if isinstance(config, dict) else None
            if isinstance(names, dict):
                try:
                    indexed = {int(key): value for key, value in names.items()}
                    names = [indexed[index] for index in range(len(indexed))]
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f'{filename} 中类别编号须从 0 连续排列。') from exc
            if names is not None:
                if not isinstance(names, list):
                    raise ValueError(f'{filename} 中 names 格式无效。')
                definitions.append(names)
        normalized = [[str(name).strip().casefold() for name in names] for names in definitions]
        if normalized and any(names != normalized[0] for names in normalized):
            raise ValueError('classes.txt 与 YAML 的类别顺序不一致，请先核对类别定义。')
        names = normalized[0] if normalized else ['left', 'right']
        self.classes_assumed = not normalized
        if len(names) != 2 or set(names) != {'left', 'right'}:
            raise ValueError('灰手套编辑器支持 left / right 两类；当前数据集类别为：' + ', '.join(names))
        self.to_internal = {index: ['left', 'right'].index(name) for index, name in enumerate(names)}
        self.to_disk = {value: key for key, value in self.to_internal.items()}
        self._metadata_expected = {path: path.read_bytes() for path in self.metadata_files}

    def _history(self):
        history = {}
        path = self.metadata / 'review.jsonl'
        if not path.is_file():
            return history
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError):
            return history
        for line in lines:
            try:
                entry = json.loads(line)
                if isinstance(entry, dict) and isinstance(entry.get('name'), str):
                    history[entry['name']] = entry
            except ValueError:
                continue
        return history

    @classmethod
    def load(cls, folder, progress=None, should_continue=lambda: True):
        dataset = cls(dataset_root(folder))
        dataset.read_classes()
        history = dataset._history()
        pairs = []
        if (dataset.root / 'images').is_dir():
            pairs.append((dataset.root / 'images', dataset.root / 'labels'))
        for split in ('train', 'val', 'valid', 'test'):
            if (dataset.root / split / 'images').is_dir():
                pairs.append((dataset.root / split / 'images', dataset.root / split / 'labels'))
        files = []
        for image_dir, label_dir in pairs:
            for path in sorted(image_dir.rglob('*')):
                if not should_continue():
                    raise InterruptedError('已取消打开数据集')
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    label = label_dir / path.relative_to(image_dir).with_suffix('.txt')
                    files.append((path, label))
        if not files:
            raise ValueError('images/ 中没有可读取的图片。')
        used_labels = set()
        all_formats = []
        for index, (path, label) in enumerate(files, 1):
            if not should_continue():
                raise InterruptedError('已取消打开数据集')
            dataset.inside(label)
            resolved = label.resolve()
            if resolved in used_labels:
                raise ValueError(f'多张图片对应同一个标签文件，无法安全编辑：{label}')
            used_labels.add(resolved)
            name = path.relative_to(dataset.root).as_posix()
            record = {'name': name, 'objects': [], 'candidates': [], 'side_pending': False,
                      'status': 'review', 'reason': '已载入 YOLO 标注，可直接修改或确认'}
            data = _read(label)
            dataset.labels[name] = label
            dataset.expected[name] = data
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    if image.getexif().get(274) in {5, 6, 7, 8}:
                        width, height = height, width
                record.update(width=width, height=height)
                objects, box_format = dataset.parse(data, width, height)
                record['objects'] = objects
                dataset.formats[name] = box_format
                if box_format:
                    all_formats.append(box_format)
                if data is None:
                    record.update(status='unlabeled', reason='尚无标签文件，可直接画框')
                elif not objects:
                    record.update(status='empty_confirmed', reason='原标签为空：无目标')
                entry = history.get(name, {})
                if entry.get('label_sha256') == _digest(data) and 'status' in entry:
                    if entry['status'] in {'review', 'unlabeled', 'confirmed', 'empty_confirmed'}:
                        record.update({key: entry[key] for key in ('status', 'reason', 'side_pending') if key in entry})
            except (OSError, ValueError, UnicodeError) as exc:
                record.update(objects=[], yolo_error=str(exc), reason=f'无法编辑：{exc}')
            dataset.records[name] = sync_candidates(record)
            if progress and (index == 1 or index % 25 == 0 or index == len(files)):
                progress(index, len(files), '读取图片与 YOLO 标签')
        if all_formats:
            dataset.default_format = max(('hbb', 'obb'), key=all_formats.count)
        return dataset

    def parse(self, data, width, height):
        objects = []
        formats = set()
        if data is None:
            return objects, None
        for line_number, line in enumerate(data.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            tokens = line.split()
            try:
                if len(tokens) not in {5, 9}:
                    raise ValueError('须为 5 列普通框或 9 列旋转框')
                disk_class = int(tokens[0])
                if disk_class not in self.to_internal:
                    raise ValueError(f'未知类别编号 {disk_class}')
                values = np.asarray([float(value) for value in tokens[1:]], dtype=float)
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                    raise ValueError('坐标须为 0 到 1 的有限数值')
                if len(tokens) == 5:
                    cx, cy, w, h = values
                    if w <= 0 or h <= 0:
                        raise ValueError('框宽高须大于 0')
                    polygon = np.array([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2],
                                        [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
                    if np.any(polygon < -1e-6) or np.any(polygon > 1+1e-6):
                        raise ValueError('框超出图片边界')
                    polygon = np.clip(polygon, 0, 1)
                    formats.add('hbb')
                else:
                    polygon = values.reshape(4, 2)
                    contour = polygon.astype(np.float32)
                    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 1e-10:
                        raise ValueError('旋转框须为有效的四边形')
                    formats.add('obb')
                objects.append({'class_id': self.to_internal[disk_class], 'score': 1.0,
                                'polygon': (polygon * [width, height]).tolist()})
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'标签第 {line_number} 行：{exc}') from exc
        if len(formats) > 1:
            raise ValueError('同一标签文件混用了普通框和旋转框，请先统一格式')
        return objects, next(iter(formats), None)

    def encode(self, record):
        name = record['name']
        box_format = self.formats.get(name) or self.default_format
        return ''.join(yolo_row({**obj, 'class_id': self.to_disk[int(obj['class_id'])]},
                                record['width'], record['height'], box_format, 2)
                       for obj in record.get('objects', [])).encode('utf-8')

    def write(self, record, data):
        """Write one label atomically; refuse stale data and back up original bytes."""
        name = record['name']
        if name not in self.labels or self.records[name].get('yolo_error'):
            raise ValueError('当前标签存在读取错误，未覆盖原文件。')
        for path, expected in self._metadata_expected.items():
            if _read(path) != expected:
                raise ValueError('数据集类别定义已在外部修改，请重新打开数据集。')
        label = self.inside(self.labels[name])
        previous = self.expected[name]
        if _read(label) != previous:
            raise ValueError('此标签已被其他程序修改，请重新打开数据集后再编辑。')
        if data != previous:
            # Ultralytics may reuse labels.cache when an edit keeps file sizes
            # unchanged. Invalidate this derived index before changing a label.
            cache = self.inside(label.parent.with_suffix('.cache'))
            if cache.is_file():
                cache.unlink()
            if previous is not None:
                backup = self.inside(self.metadata / 'backups' / label.relative_to(self.root))
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    _atomic_bytes(backup, previous)
            if data is None:
                if label.exists():
                    label.unlink()
            else:
                _atomic_bytes(label, data)
            self.expected[name] = data
        self.records[name] = copy.deepcopy(record)
        warning = ''
        try:
            history_path = self.inside(self.metadata / 'review.jsonl')
            append_record(history_path, {'name': name, 'label_sha256': _digest(data),
                                        'status': record['status'], 'reason': record.get('reason', ''),
                                        'side_pending': record.get('side_pending', False)})
        except (OSError, ValueError) as exc:
            warning = f'标签已保存，但复核状态未能记录：{exc}'
        return warning

    def export(self, output, records, box_format, split=False, progress=None,
               should_continue=lambda: True):
        selected = [copy.deepcopy(record) for record in records.values()
                    if not split or record['status'] in {'auto', 'confirmed', 'empty_confirmed'}]
        if not selected or (split and len(selected) < 2):
            raise ValueError('训练集导出至少需要确认 2 张图片。' if split else '没有可导出的图片。')
        if any(record.get('yolo_error') for record in selected):
            raise ValueError('数据集中有无法解析的标签，请修正后重新打开，再导出。')
        # Existing exporter expects flat image names. Stage hardlinks or copies,
        # keeping original filenames unless nested splits contain collisions.
        with tempfile.TemporaryDirectory(prefix='.glove_yolo_export_', dir=output) as temporary:
            source = Path(temporary)
            used = set()
            for record in selected:
                if not should_continue():
                    raise InterruptedError('已取消导出')
                original = record['name']
                alias = Path(original).name
                if Path(alias).stem in used:
                    alias = hashlib.sha256(original.encode()).hexdigest()[:16] + Path(original).suffix
                used.add(Path(alias).stem)
                try:
                    os.link((self.root / original).resolve(), source / alias)
                except OSError:
                    shutil.copy2(self.root / original, source / alias)
                record['name'] = alias
            return _write_yolo_export(source, Path(output), selected, CLASSES, box_format,
                                      split=split, progress=progress, should_continue=should_continue)
