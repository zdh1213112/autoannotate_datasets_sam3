"""YOLO import/save round trips, backups, path mapping and failure behavior."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from auto_obb_annotator.gray_glove_dataset import YoloGloveDataset, dataset_root


class YoloDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'classes.txt').write_text('left\nright\n')

    def frame(self, name='frame.jpg', label='1 0.5 0.5 0.4 0.2\n', image_dir='images', label_dir='labels'):
        image = self.root / image_dir / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (200, 100)).save(image)
        path = self.root / label_dir / Path(name).with_suffix('.txt')
        if label is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(label)
        return image, path

    def test_open_is_read_only_and_hbb_has_correct_pixels(self):
        image, label = self.frame()
        data = label.read_bytes()
        dataset = YoloGloveDataset.load(self.root / 'labels')
        record = dataset.records['images/frame.jpg']
        self.assertEqual(record['objects'][0]['class_id'], 1)
        self.assertEqual(record['objects'][0]['polygon'], [[60, 40], [140, 40], [140, 60], [60, 60]])
        self.assertEqual(label.read_bytes(), data)
        self.assertFalse((self.root / '.gray_glove_editor').exists())
        self.assertEqual(dataset_root(self.root / 'images'), self.root)

    def test_reversed_class_order_is_mapped_back_to_original_ids(self):
        (self.root / 'classes.txt').write_text('right\nleft\n')
        self.frame(label='0 0.5 0.5 0.4 0.2\n')
        dataset = YoloGloveDataset.load(self.root)
        record = dataset.records['images/frame.jpg']
        self.assertEqual(record['objects'][0]['class_id'], 1)
        self.assertTrue(dataset.encode(record).startswith(b'0 '))

    def test_save_backup_exact_undo_and_reopen_status(self):
        _, label = self.frame(label='1 0.50000000 0.5 0.4 0.2\r\n')
        dataset = YoloGloveDataset.load(self.root)
        original = label.read_bytes()
        before = copy.deepcopy(dataset.records['images/frame.jpg'])
        after = copy.deepcopy(before)
        after['objects'][0]['class_id'] = 0
        after['status'] = 'confirmed'
        dataset.write(after, dataset.encode(after))
        backup = self.root / '.gray_glove_editor/backups/labels/frame.txt'
        self.assertEqual(backup.read_bytes(), original)
        reopened = YoloGloveDataset.load(self.root)
        self.assertEqual(reopened.records['images/frame.jpg']['status'], 'confirmed')
        self.assertEqual(reopened.records['images/frame.jpg']['objects'][0]['class_id'], 0)
        dataset.write(before, original)
        self.assertEqual(label.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), original)

    def test_missing_label_can_be_created_and_undo_restores_absence(self):
        _, label = self.frame(label=None)
        dataset = YoloGloveDataset.load(self.root)
        record = dataset.records['images/frame.jpg']
        self.assertEqual(record['status'], 'unlabeled')
        dataset.write({**record, 'status': 'empty_confirmed'}, b'')
        self.assertTrue(label.is_file())
        dataset.write(record, None)
        self.assertFalse(label.exists())

    def test_obb_preserves_nine_columns_and_geometry(self):
        self.frame(label='1 0.2 0.1 0.7 0.3 0.6 0.8 0.1 0.6\n')
        dataset = YoloGloveDataset.load(self.root)
        self.assertEqual(dataset.default_format, 'obb')
        record = dataset.records['images/frame.jpg']
        result = dataset.encode(record)
        self.assertEqual(len(result.split()), 9)
        objects, fmt = dataset.parse(result, 200, 100)
        self.assertEqual(objects[0]['polygon'], record['objects'][0]['polygon'])
        self.assertEqual(objects[0]['class_id'], record['objects'][0]['class_id'])

    def test_nested_splits_with_same_filename_remain_distinct(self):
        self.frame(image_dir='images/train', label_dir='labels/train')
        self.frame(image_dir='images/val', label_dir='labels/val', label='0 0.5 0.5 0.2 0.2\n')
        dataset = YoloGloveDataset.load(self.root / 'images/train')
        self.assertEqual(len(dataset.records), 2)
        self.assertNotEqual(dataset.labels['images/train/frame.jpg'], dataset.labels['images/val/frame.jpg'])

    def test_split_first_layout_and_export_remains_self_contained(self):
        self.frame(image_dir='train/images', label_dir='train/labels')
        self.frame(image_dir='val/images', label_dir='val/labels')
        dataset = YoloGloveDataset.load(self.root / 'train/images')
        self.assertEqual(dataset.root, self.root)
        target, annotated, count = dataset.export(self.root, dataset.records, 'hbb')
        self.assertEqual((annotated, count), (2, 2))
        images = list((target / 'images').iterdir())
        self.assertEqual(len(images), 2)
        self.assertTrue(all(path.is_file() and not path.is_symlink() for path in images))
        self.assertEqual(len(list((target / 'labels').iterdir())), 2)
        self.assertEqual(len(YoloGloveDataset.load(target).records), 2)

    def test_malformed_label_is_visible_but_never_overwritten(self):
        _, label = self.frame(label='1 nan 0.5 0.2 0.2\n')
        dataset = YoloGloveDataset.load(self.root)
        record = dataset.records['images/frame.jpg']
        self.assertIn('yolo_error', record)
        original = label.read_bytes()
        with self.assertRaises(ValueError):
            dataset.write(record, b'')
        self.assertEqual(label.read_bytes(), original)

    def test_external_label_or_class_change_blocks_stale_save(self):
        _, label = self.frame()
        dataset = YoloGloveDataset.load(self.root)
        record = dataset.records['images/frame.jpg']
        label.write_text('0 0.5 0.5 0.2 0.2\n')
        with self.assertRaisesRegex(ValueError, '其他程序'):
            dataset.write(record, dataset.encode(record))
        dataset = YoloGloveDataset.load(self.root)
        (self.root / 'classes.txt').write_text('right\nleft\n')
        with self.assertRaisesRegex(ValueError, '类别定义'):
            dataset.write(dataset.records['images/frame.jpg'], b'')

    def test_write_failure_keeps_labels_and_expected_bytes(self):
        _, label = self.frame()
        dataset = YoloGloveDataset.load(self.root)
        record = copy.deepcopy(dataset.records['images/frame.jpg'])
        record['objects'][0]['class_id'] = 0
        original = label.read_bytes()
        with patch('auto_obb_annotator.gray_glove_dataset._atomic_bytes', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                dataset.write(record, dataset.encode(record))
        self.assertEqual(label.read_bytes(), original)
        self.assertEqual(dataset.expected['images/frame.jpg'], original)

    def test_cancelled_load_and_export_leave_no_artifacts(self):
        self.frame()
        with self.assertRaises(InterruptedError):
            YoloGloveDataset.load(self.root, should_continue=lambda: False)
        self.assertFalse((self.root / '.gray_glove_editor').exists())
        dataset = YoloGloveDataset.load(self.root)
        with self.assertRaises(InterruptedError):
            dataset.export(self.root, dataset.records, 'hbb', should_continue=lambda: False)
        self.assertFalse(list(self.root.glob('.glove_yolo_export_*')))
        self.assertFalse(list(self.root.glob('yolo_*')))

    def test_yaml_order_conflict_and_duplicate_label_names_rejected(self):
        self.frame()
        (self.root / 'dataset.yaml').write_text('names: [right, left]\n')
        with self.assertRaisesRegex(ValueError, '顺序不一致'):
            YoloGloveDataset.load(self.root)
        (self.root / 'dataset.yaml').write_text('names: [left, right]\n')
        self.frame(name='frame.png')
        with self.assertRaisesRegex(ValueError, '同一个标签'):
            YoloGloveDataset.load(self.root)

    def test_label_edit_invalidates_derived_yolo_cache(self):
        self.frame()
        dataset = YoloGloveDataset.load(self.root)
        cache = self.root / 'labels.cache'
        cache.write_bytes(b'stale derived labels')
        record = copy.deepcopy(dataset.records['images/frame.jpg'])
        record['objects'][0]['class_id'] = 0
        dataset.write(record, dataset.encode(record))
        self.assertFalse(cache.exists())
