import os
import argparse
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(in_dir: Path):
    return sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def ensure_bounds(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return x, y, w, h

def point_in_box(pt, box):
    (x, y) = pt
    bx, by, bw, bh = box
    return (x >= bx) and (x <= bx + bw) and (y >= by) and (y <= by + bh)

def nearest_box_center(pt, boxes):
    px, py = pt
    centers = [((bx + bw/2), (by + bh/2)) for (bx, by, bw, bh) in boxes]
    d2 = [ (cx - px)**2 + (cy - py)**2 for (cx, cy) in centers ]
    return int(np.argmin(d2))

def choose_face_via_click(img_bgr, boxes):
    """Show image with boxes; return index of chosen face via mouse click."""
    chosen = {"pt": None}
    display = img_bgr.copy()

    # Draw boxes with indices
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, f"{i}", (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    win = "Pick a face (click inside a box). Press ESC to skip."
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            chosen["pt"] = (mx, my)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if chosen["pt"] is not None:
            pt = chosen["pt"]
            # Prefer click inside a bounding box
            idx = None
            for i, b in enumerate(boxes):
                if point_in_box(pt, b):
                    idx = i
                    break
            if idx is None:
                idx = nearest_box_center(pt, boxes)
            cv2.destroyWindow(win)
            return idx
        if key == 27:  # ESC to skip this image
            cv2.destroyWindow(win)
            return None
        
def expand_box(x, y, w, h, W, H, scale=1.8, top_boost=0.25):
    """
    Turn a face box into a 'head' box.
    - scale: overall box growth (1.0 = no growth). Try 1.6–2.0.
    - top_boost: push box upward (fraction of new height) to include hair.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0

    new_size = int(scale * max(w, h))
    nx = int(cx - new_size / 2)
    ny = int(cy - new_size / 2)

    # lift the box up a bit to include hair/forehead
    ny = ny - int(top_boost * new_size)

    # clamp to image bounds
    nx = max(0, nx)
    ny = max(0, ny)
    new_size = min(new_size, W - nx, H - ny)

    return nx, ny, new_size, new_size

def main():
    parser = argparse.ArgumentParser(description="Sweep a directory for faces and save chosen crops.")
    parser.add_argument("--input_dir", type=Path, help="Directory of images")
    args = parser.parse_args()

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"Not a directory: {in_dir}")

    out_dir = Path(f"{in_dir.name}_faces")
    out_dir.mkdir(parents=True, exist_ok=True)

    # OpenCV Haar cascade (bundled with cv2)
    cascade_path = os.path.join("face_weights", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit(f"Failed to load cascade at {cascade_path}")

    images = list_images(in_dir)
    if not images:
        print("No images found.")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Skip] Could not read: {img_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tune these if needed
        boxes = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(boxes) == 0:
            print(f"[0 faces] Skip: {img_path.name}")
            continue

        H, W = img.shape[:2]

        if len(boxes) == 1:
            x, y, w, h = boxes[0]
            x, y, w, h = ensure_bounds(x, y, w, h, W, H)
            crop = img[y:y+h, x:x+w]
            save_path = out_dir / (img_path.stem + "_face.jpg")
            cv2.imwrite(str(save_path), crop)
            print(f"[1 face] Saved: {save_path.name}")
            continue

        # Multiple faces: ask user
        idx = choose_face_via_click(img, boxes)
        if idx is None:
            print(f"[Multi faces] Skipped by user: {img_path.name}")
            continue

        x, y, w, h = boxes[idx]
        x, y, w, h = ensure_bounds(x, y, w, h, W, H)
        nx, ny, nw, nh = expand_box(x, y, w, h, W, H, scale=1.8, top_boost=0.25)
        crop = img[ny:ny+nh, nx:nx+nw]
        save_path = out_dir / (img_path.stem + "_face.jpg")
        cv2.imwrite(str(save_path), crop)
        print(f"[Multi faces] Chose {idx} -> Saved: {save_path.name}")

    print(f"Done. Crops in: {out_dir}")


if __name__ == "__main__":

    main()
