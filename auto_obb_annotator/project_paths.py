from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MOBILE_SAM_MODEL = MODELS_DIR / "mobile_sam.pt"
SAM3_MODEL = Path(os.getenv("SAM3_MODEL_PATH", MODELS_DIR / "sam3.pt")).expanduser()
GROUNDING_DINO_MODEL = Path(
    os.getenv("GROUNDING_DINO_MODEL", MODELS_DIR / "groundingdino_swint_ogc_quant.onnx")
).expanduser()
GROUNDING_DINO_TOKENIZER = Path(
    os.getenv("GROUNDING_DINO_TOKENIZER", MODELS_DIR / "bert_base_uncased_tokenizer.json")
).expanduser()
REALSENSE_DATASET_DIR = DATA_DIR / "realsense_dataset"
REALSENSE_COLOR_DIR = REALSENSE_DATASET_DIR / "color"
REALSENSE_DEPTH_DIR = REALSENSE_DATASET_DIR / "depth"
REALSENSE_BG_DIR = REALSENSE_DATASET_DIR / "background"
OBB_DATASET_DIR = DATA_DIR / "obb_dataset"
HBB_DATASET_DIR = DATA_DIR / "hbb_dataset"


def display_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return str(target.relative_to(ROOT_DIR))
    except ValueError:
        return str(target)


def resolve_path(path: str | Path) -> Path:
    """将界面中的绝对/项目根目录相对路径解析为绝对路径。"""
    target = Path(path).expanduser()
    if target.is_absolute():
        return target.resolve()
    # display_path() 返回的是相对于 ROOT_DIR 的路径，不能相对于当前 cwd 解析。
    return (ROOT_DIR / target).resolve()
