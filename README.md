# SAM3 Auto OBB Annotator

基于 SAM3 / MobileSAM 的桌面自动标注工具，可导出 YOLO-OBB 旋转框或 YOLO-HBB
普通框。本项目由指定脚本 `final_auto_obb_qt2_dedup.py` 整理而成，保留了 SAM3
文本提示、模板/背景模式、特征校验和增强去重逻辑。

## 主要功能

- **SAM3 文本提示模式**：输入 `hand`、`cup` 等英文目标名称，SAM3 直接检测并分割。
- **传统模板模式**：文本提示留空时，模板匹配/背景差分生成候选，MobileSAM 分割。
- 可选 YOLO-OBB 四点格式或 YOLO-HBB `cx cy w h` 格式。
- SAM3 结果自动分配 `left` / `right` 类别。
- mask/box IoU 去重、包含小框抑制、近邻双框抑制。
- CUDA OOM 时自动降低 MobileSAM batch。
- 可视化输出和缓存结果重新过滤。

## 类别与左右手标注规则

当前程序的“左右手”是**按图像位置**分配，不是根据手掌外观判断真实左右手：

- 文本提示中包含 `hand` 或 `hands` 时，自动启用两类：`0=left`、`1=right`。
- 只有一个手时，框中心位于图像左半边标为 `left`，右半边标为 `right`。
- 检测到两只手时，横坐标更小的框标为 `left`，更大的框标为 `right`。
- 超过两个候选框时，以最左和最右候选框中心的中点作为分界。

因此，如果相机画面发生镜像、目标左右定义不是“画面左右”，需要在导出后人工检查，
或修改 `assign_left_right_classes()` 的规则。

其他物体不会再自动套用 `left/right`：

- 文本提示填写 `cup`、`bottle`、`box` 等时，默认生成单类别标签 `0`。
- 界面中的“类别名称”可以填写 `cup`、`bottle` 等，程序会把它写入 `dataset.yaml`。
- 如果类别名称保持 `object`，程序会尝试从第一个文本提示推断，例如 `cup` → `cup`。
- 一次运行建议只使用一个物体类别；如果要做 `cup`、`bottle` 多类别数据集，分别运行并
  合并数据集，或后续扩展为多文本提示到类别 ID 的映射。SAM3 返回的是目标框/掩码，
  当前代码不会自动知道同一张图中每个候选框对应哪个不同类别。

推荐示例：

| 目标 | 文本提示 | 类别名称 | 输出类别 |
|---|---|---|---|
| 双手 | `hand` | 任意（忽略） | `left`, `right` |
| 杯子 | `cup` | `cup` | `cup` |
| 瓶子 | `bottle` | `bottle` | `bottle` |
| 化妆盒 | `cosmetic box` | `cosmetic_box` | `cosmetic_box` |
- 所有数据、权重路径均使用项目相对路径或环境变量，不绑定某台电脑。

## 项目结构

```text
auto-obb-annotator/
├── auto_obb_annotator/
│   ├── app.py               # 来自 final_auto_obb_qt2_dedup.py 的主程序
│   ├── project_paths.py     # 统一管理项目路径和模型路径
│   ├── __main__.py
│   └── __init__.py
├── data/                    # 原图与导出结果，默认不进 Git
├── models/
│   ├── sam3.pt              # 约 3.45 GB，本机存在，但被 Git 忽略
│   └── mobile_sam.pt        # 传统模板模式的回退模型
├── scripts/check_setup.py
├── pyproject.toml
├── requirements.txt
└── run.py
```

## 模型文件与 GitHub 限制

SAM3 权重大小约 **3.45 GB**，而 GitHub 普通 Git 的单文件硬限制是 **100 MB**，所以
`models/sam3.pt` 已加入 `.gitignore`。当前新项目文件夹中已经复制了该文件，可以本机
直接运行，但 `git add .` 不会上传它。

### 下载 SAM3 权重

SAM3 需要手动下载核心模型权重，否则文本提示模式无法正常运行：

- 模型文件：`sam3.pt`
- ModelScope 下载地址：[facebook/sam3/sam3.pt](https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt)

可以直接下载到项目的 `models/` 目录：

```bash
wget -O models/sam3.pt \
  https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt
```

也可以使用浏览器打开上述链接下载，然后将文件放到：

```text
models/sam3.pt
```

如果权重放在其他目录，可以通过绝对路径指定：

```bash
export SAM3_MODEL_PATH=/absolute/path/to/sam3.pt
```

当前本地权重校验值：

```text
SHA256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

如确实需要通过 GitHub 分发 3.45 GB 权重，普通 Git 和免费的 Git LFS 通常都不合适，
建议使用模型官方发布地址、Hugging Face、对象存储或单独的 Release/下载说明，并确认
模型许可证允许重新分发。

## 环境安装

建议 Linux、Python 3.10/3.11、NVIDIA GPU 和 CUDA。

先根据显卡/CUDA 版本从 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)
安装合适的 PyTorch，再安装项目：

```bash
git clone https://github.com/YOUR_NAME/auto-obb-annotator.git
cd auto-obb-annotator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 按 PyTorch 官网命令安装 CUDA 版 torch
pip install -e .
```

SAM3 文本 API 要求包含 `SAM3SemanticPredictor` 和 `build_sam3_image_model` 的新版
Ultralytics。本项目依赖声明为 `ultralytics>=8.4.38`。

安装完成后检查：

```bash
python scripts/check_setup.py
```

如果希望同时验证大模型 SHA-256（需要读取整个 3.45 GB 文件，会比较慢）：

```bash
CHECK_MODEL_HASH=1 python scripts/check_setup.py
```

## 运行

```bash
python run.py
```

D435 条形码单目标数据集使用独立入口，包含限定区域、逐张实时预览和人工修正：

```bash
python run_barcode.py
```

使用步骤见 [`BARCODE_MODE.md`](BARCODE_MODE.md)。

自行设置类别、提示词和每张目标数量的通用入口：

```bash
python run_generic.py
```

使用步骤见 [`GENERIC_MODE.md`](GENERIC_MODE.md)。

灰白护具 + 黑色手指的第一视角防护手套数据，使用完整框专用入口：

```bash
python run_protective_gloves.py
```

该入口默认导出 HBB，连接分开的手指/掌面 mask、增加完整框留边，并在主窗口
逐张实时显示标注结果。详细说明见
[`PROTECTIVE_GLOVE_MODE.md`](PROTECTIVE_GLOVE_MODE.md)。
原来的纯黑手套模式仍使用 `python run_black_gloves.py`，详细说明见
[`BLACK_GLOVE_MODE.md`](BLACK_GLOVE_MODE.md)。

OpenTouch 已解码图片只标注完整可见右手、并逐帧交付 `bboxes.jsonl` 时，使用独立入口：

```bash
python run_opentouch_bboxes.py
```

该入口不生成 YOLO 标签。详细规则和输出格式见
[`OPENTOUCH_BBOX_MODE.md`](OPENTOUCH_BBOX_MODE.md)。

安装为可编辑包后也可以：

```bash
auto-obb-annotator
# 或 python -m auto_obb_annotator
```

### SAM3 文本提示模式（推荐）

1. 选择原始图片目录和独立导出目录。
2. “文本提示”填写英文目标，例如 `hand`。
3. 如果不是 hand，在“类别名称”填写训练类别，例如 `cup`。
4. 选择 OBB 或 HBB 标签格式。
5. 固定机位时勾选“只标注灰色台面区域”。默认区域对应示例中的红框；如机位不同，
   点击“重新框选台面区域”，拖框后按 `Space` 确认。
6. 点击“一键开始自动标注”。

只要文本提示非空，程序就要求 `sam3.pt` 存在并使用 SAM3；找不到时会明确报错，不会
静默切换成别的检测模型。

工作区域过滤会同时用于 SAM3、GroundingDINO 和模板模式：候选框中心必须位于区域内，
并且至少 50% 的框面积与区域相交。区域坐标按图片宽高归一化并自动保存，因而同一固定
机位即使图片分辨率改变也可继续使用。

### 模板 + MobileSAM 模式

1. 文本提示保持为空。
2. 可选背景目录。
3. 点击“提取/旋转模板”，框选几个代表性目标。
4. 点击“一键开始自动标注”。

此模式使用 `models/mobile_sam.pt`，适合不希望用文本提示或需要沿用原有流程的场景。

## 输出目录

```text
output/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
├── visualizations/
└── dataset.yaml
```

OBB 标签：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

HBB 标签：

```text
class_id center_x center_y width height
```

坐标均归一化到 `[0, 1]`。`dataset.yaml` 使用 `path: .`，整个导出目录移动后仍可使用。

## 可选 GroundingDINO 资源

代码保留了原脚本中的 GroundingDINO ONNX 类，资源路径可以用以下环境变量配置：

```bash
export GROUNDING_DINO_MODEL=/absolute/path/to/groundingdino_swint_ogc_quant.onnx
export GROUNDING_DINO_TOKENIZER=/absolute/path/to/bert_base_uncased_tokenizer.json
```

当前文本提示主流程优先且强制使用 SAM3；通常不需要配置这两个资源。


## 第三方与许可证

参见 [THIRD_PARTY.md](THIRD_PARTY.md)。权重不等同于原创程序代码许可证，请确认 SAM3、
Ultralytics、MobileSAM 及权重文件的上游许可证后再分发或商用。

本仓库暂未添加原创代码的 `LICENSE`，发布前请根据代码权属选择合适许可证。
