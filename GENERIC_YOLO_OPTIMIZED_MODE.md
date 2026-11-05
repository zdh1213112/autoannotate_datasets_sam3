# YOLO 标注器优化版

```bash
conda activate annotator
python run_generic_yolo_optimized.py
```

这是 `run_generic_yolo.py` 的独立代码副本，原入口及其默认结果目录保持可用。
优化版默认将标注写入 `data/generic_yolo_optimized_annotations`，默认权重仍为
`models/best_top_barcode_0929.pt`。若需接着处理旧项目，可在界面中将结果目录改为
原项目目录；请确认图片目录和模型权重与该项目一致。

主要交互改进：

- 默认导出 YOLO-HBB 普通框，与普通 YOLO 检测权重一致；需要 OBB 时可切换导出格式。
- 图片列表增加文件名搜索和状态筛选，可与原有“只看待复核/待重检”组合使用。
- 点击“放大查看检测图”或按 `Alt+Z`，可按原尺寸或自定义比例查看带检测框的图片。
- 检测进度显示已处理张数和预计剩余时间，状态消息与各状态图片数量分开显示。
- 主窗口内容可滚动，较低分辨率屏幕也能访问底部操作按钮。

其余检测、抽样、框选区域、人工复核和导出流程沿用原版。
