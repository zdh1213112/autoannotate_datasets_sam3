# OpenTouch 右手 BBox JSONL 模式

这个模式直接读取已经解码好的 OpenTouch 图片，只生成一个逐帧交付文件：

```text
datasets/opentouch_annotations/bboxes.jsonl
```

它不会生成 YOLO 标签、`dataset.yaml`、图片副本或 train/val 划分。原来的
`run_black_gloves.py` 不受影响。

## 安装与运行

```bash
pip install -e .
python run_opentouch_bboxes.py
```

在界面中选择 `decoded_data` 目录，设置输出文件后开始处理。输入结构为：

```text
decoded_data/
├── frames.jsonl
└── images/<source_stem>/<clip_id>/<source_frame_index>.jpg
```

程序按照 `frames.jsonl` 的顺序读取已有 JPG，并直接使用其中的 `sample_id`、
`source_file`、`clip_id`、`source_frame_index` 和 `image` 字段。不打开 HDF5，不提取、
复制或生成任何图片。

## 输出约定

每个源帧严格输出一行，并保留冗余身份字段：

```json
{"sample_id":"eat_mcdonalds::demo_00::000000","source_file":"eat_mcdonalds.hdf5","clip_id":"demo_00","source_frame_index":0,"bbox_xyxy":[312.5,184.0,521.0,431.5]}
{"sample_id":"eat_mcdonalds::demo_00::000001","source_file":"eat_mcdonalds.hdf5","clip_id":"demo_00","source_frame_index":1,"bbox_xyxy":null}
```

- 有明确且完整可见的右手：输出像素坐标 `[x1, y1, x2, y2]`；
- 左手入镜：忽略左手；
- 只有左手、右手不可用：输出 `null`；
- 左右证据冲突或存在多个可信右手候选：输出 `null`；
- 候选框接触画面边界，视为右手没有完整可见：输出 `null`；
- JPEG 损坏或无法解码：仍保留该帧的一行，但输出 `null`。

有效右手框通过上述规则后，默认会向左、右各扩展原框宽度的 `10%`，向上、下各扩展
原框高度的 `10%`，并裁剪到图像范围内，从而为手部边缘留出余量。界面中的
“BBox 每侧扩边”可以调整该比例；设为 `0%` 可恢复模型原始框。

程序使用三组 SAM3 提示（通用黑手套、右手黑手套、左手黑手套）交叉验证，不再使用
原程序的“按画面左右位置分配 left/right”规则。自动模型仍可能出现误判，正式交付前应抽检
有框样本和各类 `null` 样本；可通过界面的置信度、左右分差和边界余量提高保守程度。

三组提示在一次 SAM3 forward 中同时执行，共享同一份图像编码，不会对同一帧重复运行三次
最耗时的视觉 backbone。本机 RTX 5090 D v2 使用真实 OpenTouch 帧测试时，三次独立调用
平均为 `0.2514 秒/帧`，合并提示后平均为 `0.0983 秒/帧`，推理部分约提速 `2.56×`。

处理过程中先写 `bboxes.jsonl.partial`，全部帧写完且行数校验通过后才原子替换正式文件。
手动停止或发生异常时，部分结果会保留在 `.partial` 文件中，不会覆盖上一版正式文件。

## 精确进度与可视化

界面进度条显示已处理帧数、总帧数、两位小数百分比和按当前平均速度计算的预计剩余时间，
例如：

```text
3750 / 327030（1.15%）  预计剩余 14 小时 20 分钟
```

右侧实时预览每 5 帧刷新一次：绿色框是有效右手 bbox，红色 `NULL` 是无用样本；标题同时
显示 `sample_id` 和具体判定原因。实时预览只存在于内存/界面中，不会保存可视化图片。

运行过程中使用 `bboxes.jsonl.partial` 防止异常覆盖上一版正式结果；全部完成且行数校验通过
后，它会原子重命名为 `bboxes.jsonl`。因此正常完成后的唯一新增交付文件就是
`bboxes.jsonl`。
