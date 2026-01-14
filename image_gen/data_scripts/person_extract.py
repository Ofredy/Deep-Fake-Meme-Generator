import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import os

############################################
# IoU utility (for tracking same person)
############################################
def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2-ax1) * max(0, ay2-ay1)
    areaB = max(0, bx2-bx1) * max(0, by2-by1)

    denom = (areaA + areaB - inter_area + 1e-6)
    if denom <= 0:
        return 0.0
    return inter_area / denom

############################################
# Resize + center on 256x256 white canvas
############################################
def put_on_256_canvas(crop_bgr):
    target_size = 256
    ch, cw, _ = crop_bgr.shape
    scale = target_size / max(ch, cw)
    new_w = int(round(cw * scale))
    new_h = int(round(ch * scale))
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    off_x = (target_size - new_w) // 2
    off_y = (target_size - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    return canvas

############################################
# Take frame + mask -> white bg crop
############################################
def crop_person_on_white(frame_bgr, mask, bbox):
    h, w, _ = frame_bgr.shape
    white_bg = np.ones_like(frame_bgr, dtype=np.uint8) * 255
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    comp = np.where(mask3 == 1, frame_bgr, white_bg)

    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    crop = comp[y1:y2, x1:x2]
    return crop

############################################
# Run YOLOv8-seg and return all "person" instances
############################################
def run_person_seg(model, frame_bgr):
    """
    Returns list of dicts:
    [
      {
        "mask": (H,W) uint8 {0,1} at *full frame res*,
        "bbox": [x1,y1,x2,y2],
        "score": float
      },
      ...
    ]
    Only class == 'person' (COCO class 0).
    """
    h, w, _ = frame_bgr.shape
    results = model.predict(frame_bgr, verbose=False)[0]

    persons = []
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    clss  = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
    confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

    if results.masks is not None:
        raw_masks = results.masks.data.cpu().numpy()  # (N, Mh, Mw)
    else:
        raw_masks = []

    for det_idx, (box, c, score) in enumerate(zip(boxes, clss, confs)):
        if c != 0:
            continue  # skip non-person
        if len(raw_masks) == 0:
            continue  # no mask? skip

        mask_small = raw_masks[det_idx]  # (Mh, Mw)
        mask_big = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_big = (mask_big > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = box.astype(int)
        persons.append({
            "mask": mask_big,
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
        })

    return persons

############################################
# Single IMAGE path -> single 256x256 output
############################################
def process_image(model, image_path, out_path, autopick_biggest=False):
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image {image_path}")

    persons = run_person_seg(model, frame_bgr)
    if len(persons) == 0:
        raise RuntimeError("No people detected in image.")

    if len(persons) == 1 or autopick_biggest:
        if len(persons) > 1 and autopick_biggest:
            areas = [
                (i, (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]))
                for i, p in enumerate(persons)
            ]
            chosen_i = max(areas, key=lambda x: x[1])[0]
        else:
            chosen_i = 0
        chosen = persons[chosen_i]
    else:
        print("Multiple people detected in image:")
        for i, p in enumerate(persons):
            x1,y1,x2,y2 = p["bbox"]
            crop_prev = crop_person_on_white(frame_bgr, p["mask"], p["bbox"])
            preview_small = put_on_256_canvas(crop_prev)
            cv2.imshow(f"person_{i}", preview_small)
            print(f" index {i} | bbox {p['bbox']} | score {p['score']:.3f}")
        print("Type index of the person you want (then press Enter).")
        chosen_i = int(input().strip())
        cv2.destroyAllWindows()
        chosen = persons[chosen_i]

    person_crop = crop_person_on_white(frame_bgr, chosen["mask"], chosen["bbox"])
    canvas256 = put_on_256_canvas(person_crop)

    ok = cv2.imwrite(out_path, canvas256)
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")
    print(f"Saved {out_path}")

############################################
# Main VIDEO -> frames loop
############################################
def process_video(model, video_path, out_dir, autopick_biggest=False, iou_threshold=0.3):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    tracked_box = None
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break  # end of video

        persons = run_person_seg(model, frame_bgr)

        if len(persons) == 0:
            print(f"[frame {frame_idx}] no people found")
            frame_idx += 1
            continue

        # STEP 1: choose the target person
        if tracked_box is None:
            if autopick_biggest:
                areas = [
                    (i, (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]))
                    for i, p in enumerate(persons)
                ]
                chosen_i = max(areas, key=lambda x: x[1])[0]
                chosen = persons[chosen_i]
                tracked_box = chosen["bbox"]
                print(f"[frame {frame_idx}] autopicked person {chosen_i}")
            else:
                print(f"[frame {frame_idx}] multiple people detected:")
                for i, p in enumerate(persons):
                    crop_prev = crop_person_on_white(frame_bgr, p["mask"], p["bbox"])
                    preview_small = put_on_256_canvas(crop_prev)
                    cv2.imshow(f"person_{i}", preview_small)
                    print(f" index {i} | bbox {p['bbox']} | score {p['score']:.3f}")
                print("Type index of the person you want to track (then press Enter).")
                chosen_i = int(input().strip())
                cv2.destroyAllWindows()
                chosen = persons[chosen_i]
                tracked_box = chosen["bbox"]
        else:
            # subsequent frames: match by IoU
            best_i = None
            best_score = -1.0
            for i, p in enumerate(persons):
                score = iou(tracked_box, p["bbox"])
                if score > best_score:
                    best_score = score
                    best_i = i
            chosen = persons[best_i]
            if best_score < iou_threshold:
                print(f"[frame {frame_idx}] low IoU match ({best_score:.2f}), might be ID switch")
            tracked_box = chosen["bbox"]

        # STEP 2: make white bg crop and center to 256x256
        person_crop = crop_person_on_white(frame_bgr, chosen["mask"], chosen["bbox"])
        canvas256 = put_on_256_canvas(person_crop)

        # STEP 3: save it
        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, canvas256)
        print(f"[frame {frame_idx}] saved {out_path}")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

############################################
# CLI entry
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a single tracked person from VIDEO into 256x256 white-bg cutouts, or do the same for a SINGLE IMAGE."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="Path to input video (.mp4 etc.)")
    group.add_argument("--image_path", type=str, help="Path to a single input image.")

    parser.add_argument("--out_dir", type=str, help="Directory to save frames (video mode).")
    parser.add_argument("--out_path", type=str, help="Output image path (image mode).")

    parser.add_argument("--autopick_biggest", action="store_true",
                        help="If set, auto-pick largest person instead of asking.")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="Min IoU to consider same person across frames (video mode).")
    parser.add_argument("--yolo_weights", type=str, default="yolov8x-seg.pt",
                        help="Which YOLOv8 segmentation weights to use (e.g., yolov8s-seg.pt).")
    args = parser.parse_args()

    # Load model once and reuse
    model = YOLO(args.yolo_weights)

    if args.image_path:
        if not args.out_path:
            raise ValueError("--out_path is required when using --image_path")
        process_image(
            model=model,
            image_path=args.image_path,
            out_path=args.out_path,
            autopick_biggest=args.autopick_biggest
        )
    else:
        if not args.out_dir:
            raise ValueError("--out_dir is required when using --video_path")
        process_video(
            model=model,
            video_path=args.video_path,
            out_dir=args.out_dir,
            autopick_biggest=args.autopick_biggest,
            iou_threshold=args.iou_threshold
        )
