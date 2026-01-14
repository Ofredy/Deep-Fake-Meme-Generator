#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full preprocessing pipeline:\n"
            "raw .mp4s -> person-only frames -> cleaned videos -> 5s clips"
        )
    )
    parser.add_argument("--in_dir", type=Path, required=True, help="Input directory containing raw .mp4 files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Root output directory for all preprocessed data.")
    parser.add_argument("--fps", type=int, default=30, help="FPS for generated videos and clips (default: 30).")
    parser.add_argument("--seg", type=int, default=5, help="Segment length in seconds for final clips (default: 5).")
    parser.add_argument("--yolo_weights", type=str, default="yolov8x-seg.pt",
                        help="YOLOv8 segmentation weights passed to person_extract.py.")
    parser.add_argument("--autopick_biggest", action="store_true",
                        help="Auto-select largest person (no interactive prompt).")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if intermediate outputs already exist.")

    args = parser.parse_args()

    in_dir, out_dir = args.in_dir, args.out_dir

    if not in_dir.exists():
        raise SystemExit(f"[ERR] Input directory does not exist: {in_dir}")

    # Output structure
    frames_root = out_dir / "frames"
    videos_root = out_dir / "videos"
    clips_root = out_dir / "clips"
    for d in (frames_root, videos_root, clips_root):
        d.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    person_extract_py = script_dir / "person_extract.py"
    frames_to_video_py = script_dir / "frames_to_video.py"
    clip_mp4s_py = script_dir / "clip_mp4s.py"

    for script in (person_extract_py, frames_to_video_py, clip_mp4s_py):
        if not script.exists():
            raise SystemExit(f"[ERR] Missing script: {script}")

    mp4s = sorted(in_dir.glob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"[ERR] No .mp4 files found in {in_dir}")

    print(f"[INFO] Found {len(mp4s)} input videos in {in_dir}")
    print("[INFO] Mode:", "FORCE (overwrite)" if args.force else "RESUME (skip existing)")

    for vid_path in mp4s:
        stem = vid_path.stem
        print(f"\n=== Processing {vid_path.name} ===")

        try:
            frame_dir = frames_root / stem
            frame_dir.mkdir(parents=True, exist_ok=True)

            intermediate_video = frames_root / f"{stem}.mp4"
            cleaned_video_final = videos_root / f"{stem}.mp4"

            # Skip if already done and not forcing
            if cleaned_video_final.exists() and not args.force:
                print(f"[SKIP] Cleaned video already exists: {cleaned_video_final}")
                continue

            # STEP 1: person_extract
            frames_exist = any(frame_dir.glob("*.png"))
            if args.force or not frames_exist:
                print(f"[STEP] Running person_extract.py for {vid_path.name}")
                cmd1 = [
                    sys.executable, str(person_extract_py),
                    "--video_path", str(vid_path),
                    "--out_dir", str(frame_dir),
                    "--yolo_weights", args.yolo_weights,
                ]
                if args.autopick_biggest:
                    cmd1.append("--autopick_biggest")

                print("[CMD]", " ".join(cmd1))
                subprocess.run(cmd1, check=True)
            else:
                print(f"[SKIP] Frames already exist in {frame_dir}")

            # STEP 2: frames_to_video
            if cleaned_video_final.exists() and not args.force:
                print(f"[SKIP] Cleaned video already exists: {cleaned_video_final}")
            else:
                if intermediate_video.exists() and not args.force:
                    print(f"[INFO] Found existing intermediate: {intermediate_video}")
                else:
                    print(f"[STEP] Running frames_to_video.py for {stem}")
                    cmd2 = [
                        sys.executable, str(frames_to_video_py),
                        "--input_dir", str(frame_dir),
                        "--fps", str(args.fps),
                    ]
                    print("[CMD]", " ".join(cmd2))
                    subprocess.run(cmd2, check=True)

                if intermediate_video.exists():
                    target = videos_root / intermediate_video.name
                    if target.exists() and not args.force:
                        print(f"[SKIP] Target cleaned video already exists: {target}")
                    else:
                        print(f"[INFO] Moving {intermediate_video} -> {target}")
                        if target.exists():
                            target.unlink()
                        shutil.move(str(intermediate_video), str(target))
                else:
                    print(f"[WARN] Missing intermediate video: {intermediate_video}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed for {vid_path.name} with return code {e.returncode}")
            print("Command:", " ".join(e.cmd))
            print("Skipping to next video...")
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error while processing {vid_path.name}: {e}")
            traceback.print_exc()
            print("Skipping to next video...")
            continue

    # STEP 3: clip_mp4s.py for all cleaned videos
    print("\n=== Running final clipping step on cleaned videos ===")
    try:
        cmd3 = [
            sys.executable, str(clip_mp4s_py),
            "--in_dir", str(videos_root),
            "--out_dir", str(clips_root),
            "--fps", str(args.fps),
            "--seg", str(args.seg),
        ]
        print("[CMD]", " ".join(cmd3))
        subprocess.run(cmd3, check=True)
    except Exception as e:
        print(f"[ERROR] Final clipping step failed: {e}")
        traceback.print_exc()

    print("\n[OK] Full preprocessing complete.")
    print(f"  Frames per video: {frames_root}")
    print(f"  Cleaned person-only videos: {videos_root}")
    print(f"  Final {args.seg}s clips for training: {clips_root}")


if __name__ == "__main__":
    main()
