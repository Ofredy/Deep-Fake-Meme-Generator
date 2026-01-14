#!/usr/bin/env python3
import argparse
import os
import subprocess
import math
import shutil
from pathlib import Path

def require_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found on PATH. Install it and try again.")

def ffprobe_duration_seconds(path: Path) -> float:
    """Get duration in seconds (float) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def run_ffmpeg_segment(infile: Path, outfile_template: Path, fps: int, seg_seconds: int):
    """
    Use ffmpeg's segment muxer to split into ~5s chunks, re-encoding to ensure 30 fps.
    outfile_template should be something like: outdir / "basename#%06d#%06d.mp4"
    We'll write to a temp numeric pattern and then rename with frame indices after.
    """
    temp_pattern = outfile_template.parent / (infile.stem + "_seg_%03d.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(infile),
        "-r", str(fps),                # force output fps
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-f", "segment",
        "-segment_time", str(seg_seconds),
        "-reset_timestamps", "1",
        str(temp_pattern)
    ]
    subprocess.run(cmd, check=True)
    return list(sorted(temp_pattern.parent.glob(infile.stem + "_seg_*.mp4")))

def main():
    p = argparse.ArgumentParser(description="Clip every .mp4 in a folder into fixed 5-second, 30-fps segments.")
    p.add_argument("--in_dir", type=Path, help="Input directory containing .mp4 files")
    p.add_argument("--out_dir", type=Path, help="Output directory for segmented clips")
    p.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    p.add_argument("--seg", type=int, default=5, help="Segment length in seconds (default: 5)")
    args = p.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted([p for p in args.in_dir.glob("*.mp4") if p.is_file()])
    if not mp4s:
        print("No .mp4 files found in", args.in_dir)
        return

    frames_per_segment = args.fps * args.seg

    for src in mp4s:
        print(f"[+] Processing: {src.name}")
        # Probe total duration and total frames (assuming constant 30 fps as per your note)
        duration = ffprobe_duration_seconds(src)
        total_frames = int(round(duration * args.fps))

        # Split using segment muxer
        temp_segs = run_ffmpeg_segment(src, args.out_dir / (src.stem + "#%06d#%06d.mp4"), args.fps, args.seg)

        # Rename each temp seg to basename#startFrame#endFrame.mp4
        for idx, seg_path in enumerate(temp_segs):
            start_frame = idx * frames_per_segment
            end_frame = min(start_frame + frames_per_segment - 1, max(total_frames - 1, 0))
            # Skip empty segments if any weirdness
            if start_frame > end_frame:
                seg_path.unlink(missing_ok=True)
                continue

            out_name = f"{src.stem}#{start_frame:06d}#{end_frame:06d}.mp4"
            dst = args.out_dir / out_name

            # Final re-mux to ensure proper timestamps & fps (fast stream copy if already correct)
            # Here we copy to avoid generation loss again.
            cmd = [
                "ffmpeg", "-y",
                "-i", str(seg_path),
                "-r", str(args.fps),
                "-c:v", "copy",
                "-c:a", "copy",
                str(dst)
            ]
            subprocess.run(cmd, check=True)
            seg_path.unlink(missing_ok=True)

        print(f"    → wrote {len(list(args.out_dir.glob(src.stem + '#*.mp4')))} segments")

    print("Done.")

if __name__ == "__main__":
    main()
