import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
from config_endo import (
    FOLDER_DATA,
    RESOLUTION,
    SUBSTRING_FILTER,
)

import os
import numpy as np

# keep every N-th frame
N = 5  # adjust as needed


def downsample_npz_for_setting(substring_filter, resolution, n=N):
    base = f"{substring_filter} {resolution}"
    files = {
        "Y01":   f"{base} X Y01.npz",
        "Ym11":  f"{base} X Ym11.npz",
        "Ym11G": f"{base} X Ym11G.npz",
    }

    #C3VD_full_t1_v1_ 336 X Ym11G.npz

    for key, fname in files.items():
        path = os.path.join(FOLDER_DATA, fname)
        print(path)
        if not os.path.exists(path):
            print(f"[WARN] Skipping {key}: file not found at {path}")
            continue

        data = np.load(path)
        x = data["x"]
        Y = data["Y"]

        # keep every n-th row
        x_ds = x[::n]
        Y_ds = Y[::n]

        # copy any extra arrays (means, stds, etc.) unchanged
        out = {"x": x_ds, "Y": Y_ds}
        for k in data.files:
            if k not in ("x", "Y"):
                out[k] = data[k]

        out_path = os.path.join(FOLDER_DATA, f"{base}_step{n} X {key}.npz")
        np.savez(out_path, **out)
        print(f"[OK] Saved downsampled {key} (step={n}) to {out_path}")


if __name__ == "__main__":
    print(f"Downsampling NPZ files in {FOLDER_DATA} with step N={N}")

    for resolution in RESOLUTION:
        for substring_filter in SUBSTRING_FILTER:
            print(f"\n--- substring_filter={substring_filter}, resolution={resolution} ---")
            downsample_npz_for_setting(substring_filter, resolution, n=N)
