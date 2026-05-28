"""Download ASCAD v1 fixed-key bundle, extract ASCAD.h5."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHK = json.loads((ROOT / "leaklens" / "checksums.json").read_text())["ASCAD_data.zip"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA.mkdir(exist_ok=True)
    target = DATA / "ASCAD.h5"
    if target.exists():
        print(f"Exists: {target}  sha256={sha256(target)}")
        return 0

    zip_path = DATA / "ASCAD_data.zip"
    if not zip_path.exists():
        print(f"Downloading {CHK['url']} (~4.2 GB compressed) ...")
        urlretrieve(CHK["url"], zip_path)
        digest = sha256(zip_path)
        print(f"Downloaded: {zip_path}  sha256={digest}")
        if CHK["sha256"] != "PLACEHOLDER_VERIFY_ON_FIRST_DOWNLOAD" and digest != CHK["sha256"]:
            print(f"CHECKSUM MISMATCH expected={CHK['sha256']}")
            return 1
    else:
        print(f"Zip already present: {zip_path}")

    inner = CHK["extracted_target"]
    print(f"Extracting {inner} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    print(f"Extracted: {target}  sha256={sha256(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
