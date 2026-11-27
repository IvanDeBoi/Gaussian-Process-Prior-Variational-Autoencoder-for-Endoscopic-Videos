from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


WORKSPACE = Path(r"C:\Users\Ivan\OneDrive - Universiteit Antwerpen\01 Postdoc\04 Exp\15 GPVAE Endo")
INPUT_DIR = WORKSPACE / "Pics" / "latent_first_dataset_image_time"
OUTPUT_PATH = INPUT_DIR / "latent_image_time_all_datasets_5x2_grid.png"
PLOT_SUFFIX = "_vae_image_hpa_image_time_uncertainty_3d_pca.png"

DATASET_ORDER = [
    "c1_ascending_t1_v1",
    "c1_descending_t1_v2",
    "c1_descending_t2_v3",
    "c1_rectum_t2_v2",
    "c1_rectum_t4_v3",
    "c1_transverse2_t1_v2",
    "c2_cecum_t3_v2",
    "c2_sigmoid_t2_v2",
    # The table draft says c2_transverse1_t1_v3, but the local data/checkpoints/plots
    # currently use c2_transverse2_t1_v3.
    "c2_transverse2_t1_v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine latent-space dataset plots into a 5x2 grid.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--tile-width", type=int, default=1600)
    parser.add_argument("--gutter", type=int, default=30)
    parser.add_argument("--margin", type=int, default=30)
    parser.add_argument("--plot-suffix", default=PLOT_SUFFIX)
    parser.add_argument("--uncertainty-bar", action="store_true", help="Add one shared low-to-high uncertainty bar in the empty bottom-right grid slot.")
    parser.add_argument("--shared-legend", action="store_true", help="Add one shared marker legend in the empty bottom-right grid slot.")
    return parser.parse_args()


def crop_white_margin(image: Image.Image, pad: int = 20) -> Image.Image:
    background = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, background).convert("L")
    bbox = diff.point(lambda value: 255 if value > 8 else 0).getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(left - pad, 0)
    top = max(top - pad, 0)
    right = min(right + pad, image.width)
    bottom = min(bottom + pad, image.height)
    return image.crop((left, top, right, bottom))


def load_tile(path: Path, width: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = crop_white_margin(Image.open(path).convert("RGB"))
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def interpolate_color(left: tuple[int, int, int], right: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(left[channel] + (right[channel] - left[channel]) * t) for channel in range(3))


def uncertainty_color(t: float) -> tuple[int, int, int]:
    green = (57, 169, 53)
    yellow = (245, 221, 51)
    red = (185, 35, 24)
    if t <= 0.5:
        return interpolate_color(green, yellow, t / 0.5)
    return interpolate_color(yellow, red, (t - 0.5) / 0.5)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_uncertainty_bar(canvas: Image.Image, slot: tuple[int, int, int, int], y_fraction: float = 0.45) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = slot
    slot_w = right - left
    slot_h = bottom - top
    bar_w = round(slot_w * 0.62)
    bar_h = max(22, round(slot_h * 0.035))
    bar_x = left + (slot_w - bar_w) // 2
    bar_y = top + round(slot_h * y_fraction)

    for x_offset in range(bar_w):
        color = uncertainty_color(x_offset / max(bar_w - 1, 1))
        draw.line([(bar_x + x_offset, bar_y), (bar_x + x_offset, bar_y + bar_h)], fill=color)

    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=(40, 40, 40), width=2)

    font = load_font(max(32, round(slot_h * 0.052)))
    label_y = bar_y + bar_h + max(8, round(slot_h * 0.012))
    low = "Low uncertainty"
    high = "High uncertainty"
    high_bbox = draw.textbbox((0, 0), high, font=font)
    draw.text((bar_x, label_y), low, fill=(20, 20, 20), font=font)
    draw.text((bar_x + bar_w - (high_bbox[2] - high_bbox[0]), label_y), high, fill=(20, 20, 20), font=font)


def draw_marker_legend(canvas: Image.Image, slot: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = slot
    slot_w = right - left
    slot_h = bottom - top
    title_font = load_font(max(40, round(slot_h * 0.070)))
    font = load_font(max(34, round(slot_h * 0.058)))
    x0 = left + round(slot_w * 0.16)
    y0 = top + round(slot_h * 0.12)
    line_gap = max(52, round(slot_h * 0.095))
    marker_x = x0
    text_x = x0 + round(slot_w * 0.055)
    marker_r = max(11, round(slot_h * 0.018))
    square_r = max(15, round(slot_h * 0.026))

    draw.text((x0, y0 - line_gap), "Shared legend", fill=(20, 20, 20), font=title_font)

    entries = [
        ("circle", (80, 170, 120), "Image-encoded train: coloured by frame index"),
        ("circle", (0, 0, 0), "VAE image-encoded test"),
        ("circle", (57, 169, 53), "GPVAE_SPA image-encoded test"),
        ("circle", (150, 80, 160), "GPVAE_SPA time-encoded train"),
        ("square", (57, 169, 53), "GPVAE_SPA time-encoded test"),
    ]
    for idx, (shape, fill, label) in enumerate(entries):
        cy = y0 + idx * line_gap
        if shape == "square":
            draw.rectangle(
                (marker_x - square_r, cy - square_r, marker_x + square_r, cy + square_r),
                fill=fill,
                outline=(35, 35, 35),
                width=2,
            )
        else:
            draw.ellipse(
                (marker_x - marker_r, cy - marker_r, marker_x + marker_r, cy + marker_r),
                fill=fill,
                outline=(35, 35, 35),
                width=2,
            )
        draw.text((text_x, cy - marker_r - 4), label, fill=(20, 20, 20), font=font)


def main() -> None:
    args = parse_args()
    tiles = [load_tile(args.input_dir / f"{dataset}{args.plot_suffix}", args.tile_width) for dataset in DATASET_ORDER]

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    n_cols = 2
    n_rows = 5
    canvas_w = args.margin * 2 + tile_w * n_cols + args.gutter * (n_cols - 1)
    canvas_h = args.margin * 2 + tile_h * n_rows + args.gutter * (n_rows - 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, n_cols)
        x = args.margin + col * (tile_w + args.gutter) + (tile_w - tile.width) // 2
        y = args.margin + row * (tile_h + args.gutter) + (tile_h - tile.height) // 2
        canvas.paste(tile, (x, y))

    if args.uncertainty_bar or args.shared_legend:
        empty_col = 1
        empty_row = 4
        slot_left = args.margin + empty_col * (tile_w + args.gutter)
        slot_top = args.margin + empty_row * (tile_h + args.gutter)
        slot = (slot_left, slot_top, slot_left + tile_w, slot_top + tile_h)
        if args.shared_legend:
            draw_marker_legend(canvas, slot)
        if args.uncertainty_bar:
            draw_uncertainty_bar(canvas, slot, y_fraction=0.75 if args.shared_legend else 0.45)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
