# download_google_images.py
import argparse
import hashlib
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageOps
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# ---------- utils ----------
def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s.strip().lower())

def pil_load_rgb(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return im.convert("RGB")

def center_crop_resize(im: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def sha1_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def split_list(items, train_p=0.8, val_p=0.1, test_p=0.1):
    n = len(items)
    n_train = int(round(train_p * n))
    n_val = int(round(val_p * n))
    n_test = max(0, n - n_train - n_val)
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:n_train+n_val+n_test]

def ensure_split_dirs(root: Path, cls: str):
    d = {
        "train": root / cls / "train",
        "val": root / cls / "val",
        "test": root / cls / "test",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

# ---------- crawler wrappers ----------
def crawl_google(query: str, root: Path, max_num: int, **kw):
    # parser errors can happen; keep threads low + retry implicitly via icrawler
    crawler = GoogleImageCrawler(
        storage={"root_dir": str(root)},
        parser_threads=1, downloader_threads=1,
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=None,  # or (256,256) if you want to filter thumbnails
        file_idx_offset=0,
        # icrawler uses requests; we can pass per-request kwargs via downloader_kwargs
        # but Google blocks often—keep it simple here
    )

def crawl_bing(query: str, root: Path, max_num: int, **kw):
    crawler = BingImageCrawler(storage={"root_dir": str(root)}, parser_threads=2, downloader_threads=2)
    crawler.crawl(keyword=query, max_num=max_num, min_size=None)

ENGINE_FUNCS = {
    "google": crawl_google,
    "bing": crawl_bing,
}

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Multi-engine image downloader + split/resizer")
    ap.add_argument("-s", "--search", nargs="+", required=True, help="One or more search terms (each becomes a class)")
    ap.add_argument("--data_root", type=str, required=True, help="Output root (will create <root>/<class>/{train,val,test})")
    ap.add_argument("-n", "--num", type=int, default=500, help="Target images per class AFTER dedup (default 500)")
    ap.add_argument("--img_size", type=int, default=256, help="Square resize (default 256)")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/val/test split (default 0.8 0.1 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    ap.add_argument("--engines", nargs="+", default=["google", "bing"], choices=["google", "bing"], help="Engines to use")
    ap.add_argument("--overfetch_factor", type=float, default=3.0, help="Fetch this multiple of --num per engine before dedup (default 3x)")
    args = ap.parse_args()

    data_root = Path(args.data_root); data_root.mkdir(parents=True, exist_ok=True)
    train_p, val_p, test_p = args.split
    assert abs(train_p + val_p + test_p - 1.0) < 1e-6, "Split must sum to 1.0"

    rng = random.Random(args.seed)

    for term in args.search:
        cls = safe_name(term)
        print(f"\n=== Building class '{cls}' from query: {term!r} ===")
        split_dirs = ensure_split_dirs(data_root, cls)

        # temp pool for raw downloads (multiple engines)
        with tempfile.TemporaryDirectory(prefix=f"dl_{cls}_") as td:
            tmp_root = Path(td)

            per_engine_target = int(max(1, args.num * args.overfetch_factor))
            print(f"Will try ~{per_engine_target} per engine x {len(args.engines)} engines (overfetch={args.overfetch_factor}x)")

            for eng in args.engines:
                eng_dir = tmp_root / eng; eng_dir.mkdir(parents=True, exist_ok=True)
                print(f"  -> Crawling {eng} …")
                try:
                    ENGINE_FUNCS[eng](term, eng_dir, per_engine_target)
                except Exception as e:
                    print(f"  !! {eng} crawl error: {e}")

            # gather all downloaded files
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            raw_files = [p for p in tmp_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            print(f"Downloaded {len(raw_files)} raw files before validation/dedup")

            # validate + dedupe
            seen = set()
            valid = []
            for p in raw_files:
                try:
                    _ = pil_load_rgb(p)
                    h = sha1_of_file(p)
                    if h not in seen:
                        seen.add(h)
                        valid.append(p)
                except Exception:
                    continue

            print(f"Kept {len(valid)} after dedup/corrupt filtering")
            if not valid:
                print("No valid images. Moving on.")
                continue

            # cap to requested num (post-dedup)
            rng.shuffle(valid)
            if len(valid) > args.num:
                valid = valid[:args.num]

            # split
            train_items, val_items, test_items = split_list(valid, train_p, val_p, test_p)

            def save_batch(items, dest: Path):
                count = 0
                for i, src in enumerate(items):
                    try:
                        im = pil_load_rgb(src)
                        im = center_crop_resize(im, args.img_size)
                        out = dest / f"img_{i:06d}.jpg"
                        im.save(out, format="JPEG", quality=95, optimize=True)
                        count += 1
                    except Exception:
                        continue
                return count

            print(f"Processing & splitting into {data_root/cls} @ {args.img_size}px ...")
            ntr = save_batch(train_items, split_dirs["train"])
            nva = save_batch(val_items, split_dirs["val"])
            nte = save_batch(test_items, split_dirs["test"])

            print(f"Saved {ntr} train, {nva} val, {nte} test images for class '{cls}'.")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
