# YOLO 检测权重版通用标注器

```bash
conda activate annotator
python run_generic_yolo.py
```

这是独立入口，默认使用 `models/best_top_barcode_0929.pt`，结果目录默认是
`data/generic_yolo_annotations`；原来的 `run_generic.py` 继续使用 SAM3。

界面中的“检测模型权重”可浏览选择其他 **Ultralytics YOLO 普通目标检测** `.pt` 文件。
模型类别由权重自动读取，不需要填写 SAM3 文本提示词。当前默认权重只有 `barcode` 类别。
选择其他权重后按 `F5` 会根据新配置重检，并备份旧项目配置与复核记录。

图片抽样、区域选择、固定数量和选择规则、候选框点击切换、人工复核、删除标签与
YOLO-OBB / YOLO-HBB 导出等操作与 `run_generic.py` 相同。普通 YOLO 检测框是水平矩形；
选择 OBB 导出时会把矩形写成四角坐标，不会凭空产生旋转角度。
