import argparse
from pathlib import Path

import imageio.v2 as iio
from moviepy.editor import VideoFileClip


def extract_frames_and_audio(video_path):
    video_path = Path(video_path).resolve()
    out_dir = Path.cwd() / video_path.stem   # folder = video name only, in current dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as clip:
        # Save frames
        print("[Frames] Extracting frames...")
        for i, frame in enumerate(clip.iter_frames()):
            fname = frames_dir / f"frame_{i:05d}.jpg"
            iio.imwrite(fname, frame, format="jpg")
        print(f"[Frames] Done. Saved {i+1} frames to {frames_dir}/")

        # Save audio as WAV (if it exists)
        if clip.audio is not None:
            audio_path = out_dir / f"{video_path.stem}.wav"
            print(f"[Audio] Extracting audio to {audio_path} ...")
            clip.audio.write_audiofile(str(audio_path), fps=44100, codec="pcm_s16le", logger=None)
            print("[Audio] Done.")
        else:
            print("[Audio] No audio track found; skipping.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract frames and audio from a video")
    parser.add_argument("--video", type=str, help="Path to the input .mp4 video")
    args = parser.parse_args()

    extract_frames_and_audio(args.video)
