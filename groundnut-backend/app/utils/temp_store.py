# app/utils/temp_store.py
import os, json, time, glob
from typing import Optional, Dict, Any
from app.core.config import Config

VALID_EXTS = ("jpg","jpeg","png","webp","bmp","tif","tiff")

def ensure_temp_dir():
    os.makedirs(Config.TEMP_DIR, exist_ok=True)

def _now_ts() -> int:
    return int(time.time())

def meta_path(analysis_id: str) -> str:
    return os.path.join(Config.TEMP_DIR, f"{analysis_id}.json")

def image_glob(analysis_id: str) -> str:
    return os.path.join(Config.TEMP_DIR, f"{analysis_id}.*")

def find_image_path(analysis_id: str) -> Optional[str]:
    # cari file gambar dengan ekstensi umum
    for ext in VALID_EXTS:
        p = os.path.join(Config.TEMP_DIR, f"{analysis_id}.{ext}")
        if os.path.exists(p):
            return p
    # fallback: glob apapun
    cand = [p for p in glob.glob(image_glob(analysis_id)) if not p.endswith(".json")]
    return cand[0] if cand else None

def write_temp_image(analysis_id: str, image_bytes: bytes, ext: str = "jpg") -> str:
    ensure_temp_dir()
    ext = ext.lower().lstrip(".")
    if ext not in VALID_EXTS:
        ext = "jpg"
    p = os.path.join(Config.TEMP_DIR, f"{analysis_id}.{ext}")
    with open(p, "wb") as f:
        f.write(image_bytes)
    return p

def write_meta(analysis_id: str, meta: Dict[str, Any]) -> str:
    ensure_temp_dir()
    meta = dict(meta)
    meta.setdefault("analysis_id", analysis_id)
    meta.setdefault("created_at", _now_ts())
    mp = meta_path(analysis_id)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return mp

def read_meta(analysis_id: str) -> Dict[str, Any]:
    mp = meta_path(analysis_id)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"meta not found for analysis_id={analysis_id}")
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)

def is_expired(meta: Dict[str, Any]) -> bool:
    created = int(meta.get("created_at", 0))
    return (_now_ts() - created) > int(Config.TEMP_TTL_SECONDS)

def read_image_bytes(analysis_id: str) -> bytes:
    p = find_image_path(analysis_id)
    if not p:
        raise FileNotFoundError(f"image not found for analysis_id={analysis_id}")
    with open(p, "rb") as f:
        return f.read()

def delete_bundle(analysis_id: str):
    """
    Hapus semua file terkait analysis_id di TEMP_DIR.
    Dibuat retry karena di Windows kadang file masih 'locked' sebentar.
    """
    import time
    import glob
    import os

    pattern = os.path.join(Config.TEMP_DIR, f"{analysis_id}*")
    paths = glob.glob(pattern)

    failed = []
    for p in paths:
        # retry beberapa kali
        ok = False
        for _ in range(6):  # ~ total 0.6-1.2 detik
            try:
                if os.path.isdir(p):
                    # kalau ada folder (jarang), hapus folder
                    import shutil
                    shutil.rmtree(p, ignore_errors=False)
                else:
                    os.remove(p)
                ok = True
                break
            except PermissionError:
                time.sleep(0.2)
            except FileNotFoundError:
                ok = True
                break
            except Exception:
                time.sleep(0.2)

        if not ok:
            failed.append(p)

    if failed:
        # jangan disilent lagi, biar ketahuan di console/log
        raise RuntimeError(f"Gagal menghapus temp files: {failed}")

def cleanup_expired():
    """
    Optional: panggil kadang-kadang untuk bersih-bersih.
    """
    ensure_temp_dir()
    for mp in glob.glob(os.path.join(Config.TEMP_DIR, "*.json")):
        try:
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if is_expired(meta):
                aid = os.path.splitext(os.path.basename(mp))[0]
                delete_bundle(aid)
        except Exception:
            continue
