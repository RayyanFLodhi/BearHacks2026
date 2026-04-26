from pathlib import Path


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_stem(path):
    return Path(path).stem.replace(" ", "_")


def list_image_files(folder):
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    base = Path(folder)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
