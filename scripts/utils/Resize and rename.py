import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
from pathlib import Path
from PIL import Image

from config_endo import (
    SUBSTRING_FILTER
)

input_dir = Path(f"C:/Endo/C3VDv2/{SUBSTRING_FILTER}/rgb")
output_dir = Path(f"C:/Endo/Images from datasets/C3VDv2/{SUBSTRING_FILTER}")
prefix = f'{SUBSTRING_FILTER}_336_'



# input_dir = Path("C:/Users/Ivan/OneDrive - Universiteit Antwerpen/01 Postdoc/04 Exp/15 GPVAE Endo/Utils/Cam_Calib/cfhq190l_10x10mm_checkerboard_images")
# output_dir = Path("C:/Users/Ivan/OneDrive - Universiteit Antwerpen/01 Postdoc/04 Exp/15 GPVAE Endo/Utils/Cam_Calib/cfhq190l_10x10mm_checkerboard_images - Resized")
# prefix = ''

output_dir.mkdir(parents=True, exist_ok=True)

def process_image(src: Path, dst: Path):
    print(f"Processing {src} -> {dst}")
    img = Image.open(src).convert("RGB")
    w, h = img.size
    target = 336

    # Resize so height becomes 336, preserving aspect ratio
    new_h = target
    new_w = round(w * target / h)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop width to 336
    left = (new_w - target) // 2
    img = img.crop((left, 0, left + target, target))

    img.save(dst)

for src in sorted(input_dir.glob("*.png")):
    frame = src.stem  # e.g. 0001
    dst = output_dir / f"{prefix}{frame}.png"
    process_image(src, dst)

print(f"Processed {len(list(input_dir.glob('*.png')))} images.")