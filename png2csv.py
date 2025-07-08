import argparse
import base64
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ---------- 帮助函数 ----------------------------------------------------------
def img_to_base64(p: Path) -> str:
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def extract_int_idx(path: Path) -> int:
    """
    从文件名中提取最后一段数字并返回整数。
    例:  image_007_DN → 7
          15_DN       → 15
    """
    m = re.findall(r"\d+", path.stem)
    if not m:
        raise ValueError(f"文件名 {path.name} 中未找到数字索引")
    return int(m[-1])           # 取最后一串数字


# ---------- 主流程 ------------------------------------------------------------
def main(img_dir: Path, out_csv: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    # 只保留 *_DN.xxx，并提取数字后排序
    imgs = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in exts and p.stem.upper().endswith("_DN")
    ]
    assert imgs, f"{img_dir} 内未找到 *_DN 图像！"

    imgs.sort(key=extract_int_idx)            # ← 关键：按数字排序

    ids, blocks = [], []
    for p in imgs:
        idx = extract_int_idx(p)              # 用数字索引当作 ID
        ids.append(idx)
        blocks.append(img_to_base64(p))
        print(f"[{idx:04d}] {p.name}  已编码")

    df = pd.DataFrame({"ID": ids, "BLOCK": blocks})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ 已生成: {out_csv.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 *_DN 图像批量转成 SubmitSrgb.csv")
    parser.add_argument("--img_dir", required=True, help="去噪后图像所在目录")
    parser.add_argument("--out_csv", default="./SubmitSrgb.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    main(Path(args.img_dir), Path(args.out_csv))
