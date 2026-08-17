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

其他用户克隆仓库后需要自行获得合法的 SAM3 权重并放到：

```text
models/sam3.pt
```

也可以放在任意位置，通过环境变量指定：

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

安装为可编辑包后也可以：

```bash
auto-obb-annotator
# 或 python -m auto_obb_annotator
```

### SAM3 文本提示模式（推荐）

1. 选择原始图片目录和独立导出目录。
2. “文本提示”填写英文目标，例如 `hand`。
3. 选择 OBB 或 HBB 标签格式。
4. 点击“一键开始自动标注”。

只要文本提示非空，程序就要求 `sam3.pt` 存在并使用 SAM3；找不到时会明确报错，不会
静默切换成别的检测模型。

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

