from __future__ import annotations

import hashlib
import os
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
MOBILE_SAM_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"


def check_distribution(name: str) -> bool:
    try:
        print(f"[OK] {name} {metadata.version(name)}")
        return True
    except metadata.PackageNotFoundError:
        print(f"[MISSING] {name}")
        return False


def check_model(label: str, path: Path, expected_sha256: str, required: bool) -> bool:
    if not path.is_file():
        status = "MISSING" if required else "OPTIONAL"
        print(f"[{status}] {label}: {path}")
        return not required
    size_gib = path.stat().st_size / 1024**3
    print(f"[OK] {label}: {path} ({size_gib:.2f} GiB)")
    if os.getenv("CHECK_MODEL_HASH") == "1":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            print(f"[WARN] {label} checksum differs: {digest}")
        else:
            print(f"[OK] {label} checksum")
    return True


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    ok = all(check_distribution(name) for name in (
        "numpy", "opencv-python", "Pillow", "PyQt5", "timm", "torch", "ultralytics"
    ))

    try:
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model  # noqa: F401
        from ultralytics.models.sam.predict import SAM3SemanticPredictor  # noqa: F401
        print("[OK] Ultralytics SAM3 API")
    except Exception as exc:
        print(f"[MISSING] Ultralytics SAM3 API: {exc}")
        ok = False

    sam3 = Path(os.getenv("SAM3_MODEL_PATH", ROOT / "models" / "sam3.pt")).expanduser()
    ok = check_model("SAM3", sam3, SAM3_SHA256, required=True) and ok
    check_model(
        "MobileSAM", ROOT / "models" / "mobile_sam.pt", MOBILE_SAM_SHA256, required=False
    )

    if not ok:
        print("\nSetup is incomplete. See README.md for installation instructions.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

