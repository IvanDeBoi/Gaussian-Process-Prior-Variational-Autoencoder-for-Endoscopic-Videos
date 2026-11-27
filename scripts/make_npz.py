"""Turn a video file (or a folder of frame images) into the NPZ data file the training reads.

    python scripts/make_npz.py --video V.mp4 --name kvasir_8a00709108cd4e2b --first 1000 --last 2999 --out DIR
    python scripts/make_npz.py --frames C3VD/c1_cecum_t3_v3/rgb --name c1_cecum_t3_v3 --out DIR

It does what the original two-step preparation did, in one go:
1. frames: read the video's frames --first..--last (0-based, inclusive; frame numbers are kept), or the
   images in --frames sorted by the number at the end of their name;
2. resize each frame to 336 x 336 the way `scripts/utils/Resize and rename.py` did for C3VD (scale so the
   shorter side is 336, LANCZOS, then crop the centre); frames that already are 336 x 336 are kept as they are;
3. compute the mask (DUCKNet specular highlights) and normalise the colours with
   `scripts/prepare_dataset.py`'s own `load_and_prepare_data` (unchanged, so the result is the same as before);
4. write "<out>/<name> 336 X Ym01sGM.npz" (the file the training reads); --all-normalisations also writes the
   two other files the original wrote (Y01M, Ym0s1M), which the training does not use.

At most --max-frames frames are used (default 2000, as `prepare_dataset.py`). DUCKNet needs
weights/Duck16mod_Spec_99_new.pt (or Models/Weights/); it uses the GPU if there is one. Mask previews (every
10th frame) go to Resulting_Masks/<name>_336/ in the current folder.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

RES = 336
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_336(img_bgr: np.ndarray) -> np.ndarray:
    """336 x 336 BGR: unchanged if already that size, else shorter side -> 336 (LANCZOS) and centre crop."""
    h, w = img_bgr.shape[:2]
    if (h, w) == (RES, RES):
        return img_bgr
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    scale = RES / min(h, w)
    new_w, new_h = max(RES, round(w * scale)), max(RES, round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left, top = (new_w - RES) // 2, (new_h - RES) // 2
    img = img.crop((left, top, left + RES, top + RES))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def frames_from_video(path: Path, first: int, last: int | None, max_frames: int):
    """Yield (frame number, BGR frame), read sequentially (seeking is not frame-exact for every codec)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{path.name}: {total} frames, {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {cap.get(cv2.CAP_PROP_FPS):.2f} fps")
    idx, n = 0, 0
    try:
        while n < max_frames and (last is None or idx <= last):
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= first:
                yield idx, frame
                n += 1
            idx += 1
    finally:
        cap.release()
    if n == 0:
        raise ValueError(f"no frames read from {path} in the range {first}..{last}")


def frames_from_folder(folder: Path, max_frames: int):
    """Yield (number at the end of the file name, BGR image), sorted by that number."""
    trailing = re.compile(r"(\d+)\D*$")
    files = []
    for p in folder.iterdir():
        m = trailing.search(p.stem)
        if p.suffix.lower() in IMAGE_SUFFIXES and m:
            files.append((int(m.group(1)), p))
    if not files:
        raise ValueError(f"no numbered images in {folder}")
    for num, p in sorted(files)[:max_frames]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {p}")
        yield num, img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="video file (mp4, avi, ...)")
    src.add_argument("--frames", type=Path, help="folder of numbered frame images")
    ap.add_argument("--name", required=True, help="video name used in the NPZ file name (no spaces or commas)")
    ap.add_argument("--out", type=Path, required=True, help="folder for the NPZ file")
    ap.add_argument("--first", type=int, default=0, help="first video frame, 0-based (video only)")
    ap.add_argument("--last", type=int, default=None, help="last video frame, inclusive (video only)")
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--all-normalisations", action="store_true", help="also write the Y01M and Ym0s1M files")
    ap.add_argument("--force", action="store_true", help="overwrite an existing NPZ")
    args = ap.parse_args(argv)

    if re.search(r"[\s,]", args.name):
        ap.error("--name must not contain spaces or commas")
    target = args.out / f"{args.name} {RES} X Ym01sGM.npz"
    if target.exists() and not args.force:
        print(f"{target} exists; use --force to make it again")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tmp = args.out / f".frames_{args.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        it = (frames_from_video(args.video, args.first, args.last, args.max_frames) if args.video
              else frames_from_folder(args.frames, args.max_frames))
        n = 0
        for num, frame in it:
            # name pattern that load_and_prepare_data looks for: "<name>_336_<number>.png"
            cv2.imwrite(str(tmp / f"{args.name}_{RES}_{num:06d}.png"), to_336(frame))
            n += 1
            if n % 200 == 0:
                print(f"  {n} frames written ({time.time() - t0:.0f} s)")
        print(f"{n} frames at {RES}x{RES} ({time.time() - t0:.0f} s); computing masks ...")

        from config_endo import GASTRONET_MEAN, GASTRONET_STD
        from scripts.prepare_dataset import load_and_prepare_data

        Y01, Ym0s1, Ym0s1G, x, o_mean, o_std, M = load_and_prepare_data(
            str(tmp), args.name, RES, must_preserve_color=True, max_frames=args.max_frames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    base = args.out / f"{args.name} {RES} X"
    np.savez(f"{base} Ym01sGM.npz", x=x, Y=Ym0s1G, M=M, means=GASTRONET_MEAN, stds=GASTRONET_STD)
    print(f"Saved {base} Ym01sGM.npz")
    if args.all_normalisations:
        np.savez(f"{base} Y01M.npz", x=x, Y=Y01, M=M)
        np.savez(f"{base} Ym0s1M.npz", x=x, Y=Ym0s1, M=M, means=o_mean, stds=o_std)
        print(f"Saved {base} Y01M.npz and {base} Ym0s1M.npz")
    print(f"{len(x)} frames (numbers {int(x.min())}..{int(x.max())}), valid pixels {M.mean():.4f}, "
          f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
