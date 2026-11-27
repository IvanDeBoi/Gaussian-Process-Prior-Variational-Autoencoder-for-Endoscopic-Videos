from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(r"C:\Users\Ivan\OneDrive - Universiteit Antwerpen\01 Postdoc\04 Exp\15 GPVAE Endo")
RAW_ROOT = Path(r"C:\Endo\Images from datasets\C3VDv2")
MASK_ROOT = WORKSPACE / "Resulting_Masks"
RECON_ROOT = Path(r"C:\Endo\Full train with mask\Recon")

DEFAULT_DATASET_ORDER = [
    "c1_ascending_t1_v1",
    "c1_descending_t1_v2",
    "c1_descending_t2_v3",
    "c1_rectum_t2_v2",
    "c1_rectum_t4_v3",
    "c1_transverse2_t1_v2",
    "c2_cecum_t3_v2",
    "c2_sigmoid_t2_v2",
    "c2_transverse2_t1_v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a raw/masked/VAE/GPVAE_HPA grid for configured C3VDv2 datasets."
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--recon-root", type=Path, default=RECON_ROOT)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=["VAE", "GPVAE_HPA"])
    parser.add_argument("--encoder", default="EndoVAEEnc")
    parser.add_argument("--decoder", default="EndoVAEDec")
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--lpips-resolution", type=int, default=256)
    parser.add_argument("--teacher-forcing", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "Pics" / "frame100_raw_mask_vae_gpvaehpa_diff_EndoVAEEnc_EndoVAEDec.png",
    )
    return parser.parse_args()


def configured_datasets(config_dir: Path) -> list[str]:
    datasets = []
    for config_path in sorted(config_dir.glob("config_*.py")):
        text = config_path.read_text(encoding="utf-8")
        match = re.search(r"^SUBSTRING_FILTER\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if match:
            datasets.append(match.group(1))

    ordered = [dataset for dataset in DEFAULT_DATASET_ORDER if dataset in datasets]
    ordered.extend(dataset for dataset in datasets if dataset not in ordered)
    return ordered


def find_raw_frame(raw_root: Path, dataset: str, resolution: int, frame: int) -> Path:
    frame4 = f"{frame:04d}"
    candidates = [
        raw_root / dataset / f"{dataset}_{resolution}_{frame4}.png",
        raw_root / dataset / f"{dataset}_{resolution}_{frame}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    hits = sorted((raw_root / dataset).glob(f"*{frame4}*.png"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No raw frame {frame} found for {dataset}")


def find_masked_frame(mask_root: Path, dataset: str, resolution: int, frame: int) -> Path:
    frame6 = f"{frame:06d}"
    mask_dir = mask_root / f"{dataset}_{resolution}"
    candidate = mask_dir / f"check_{frame6}.png"
    if candidate.exists():
        return candidate

    hits = sorted(mask_dir.glob(f"*{frame6}*.png"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No masked frame {frame} found for {dataset}")


def find_reconstruction(
    recon_root: Path,
    dataset: str,
    frame: int,
    model: str,
    encoder: str,
    decoder: str,
    lpips_resolution: int,
    teacher_forcing: int,
) -> Path:
    frame6 = f"{frame:06d}"
    frame_dir = recon_root / dataset / "Frames"
    preferred = (
        frame_dir
        / (
            f"{dataset} {model} {encoder} {decoder} CAUCHY 256D "
            f"L_True_{lpips_resolution} M_True G_False GAP_False_0 "
            f"TF_{teacher_forcing} {frame6}.png"
        )
    )
    if preferred.exists():
        return preferred

    pattern = f"*{model}*{encoder}*{decoder}*{frame6}.png"
    hits = sorted(path for path in frame_dir.glob(pattern) if f"TF_{teacher_forcing}" in path.name)
    if hits:
        return hits[0]

    hits = sorted(frame_dir.glob(pattern))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No {model} {encoder} reconstruction frame {frame} found for {dataset}")


def load_font(candidates: list[Path], size: int) -> ImageFont.ImageFont:
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fit_image(path: Path, cell_width: int, cell_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_width, cell_height), (0, 0, 0))
    canvas.paste(image, ((cell_width - image.width) // 2, (cell_height - image.height) // 2))
    return canvas


def difference_image(
    first_path: Path, second_path: Path, cell_width: int, cell_height: int
) -> tuple[Image.Image, float]:
    first = fit_image(first_path, cell_width, cell_height)
    second = fit_image(second_path, cell_width, cell_height)
    first_pixels = first.load()
    second_pixels = second.load()
    values = []

    for y in range(cell_height):
        for x in range(cell_width):
            r1, g1, b1 = first_pixels[x, y]
            r2, g2, b2 = second_pixels[x, y]
            values.append((abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)) / 3)

    scale = percentile(values, 98)
    if scale <= 0:
        scale = max(values) or 1

    heatmap = Image.new("RGB", (cell_width, cell_height), (0, 0, 0))
    heatmap_pixels = heatmap.load()
    index = 0
    for y in range(cell_height):
        for x in range(cell_width):
            normalized = min(values[index] / scale, 1.0)
            heatmap_pixels[x, y] = viridis(normalized ** 0.75)
            index += 1

    return heatmap, scale


def draw_colorbar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    scale: float,
    font: ImageFont.ImageFont,
) -> None:
    for offset in range(height):
        value = 1.0 - offset / max(height - 1, 1)
        color = viridis(value)
        draw.rectangle((x, y + offset, x + width - 1, y + offset), fill=color)

    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(210, 210, 210))
    draw.text((x + width + 6, y - 2), f"{scale:.1f}", fill=(25, 25, 25), font=font)
    draw.text((x + width + 6, y + height - text_height(draw, "0", font) + 2), "0", fill=(25, 25, 25), font=font)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value / 100)
    return ordered[index]


def viridis(value: float) -> tuple[int, int, int]:
    # Compact interpolation through representative Viridis control points.
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    value = max(0.0, min(1.0, value))
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if value <= right_value:
            fraction = (value - left_value) / (right_value - left_value)
            return tuple(
                round(left_color[channel] + fraction * (right_color[channel] - left_color[channel]))
                for channel in range(3)
            )
    return stops[-1][1]


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def main() -> None:
    args = parse_args()
    datasets = configured_datasets(args.workspace / "configs")
    frame4 = f"{args.frame:04d}"

    cell_width = args.resolution
    cell_height = args.resolution
    label_width = 250
    colorbar_width = 18
    colorbar_label_width = 48
    header_height = 0
    gap = 8

    font_paths = [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\segoeui.ttf")]
    font = load_font(font_paths, 18)
    colorbar_font = load_font(font_paths, 14)

    rows = []
    for dataset in datasets:
        recon_paths = [
            find_reconstruction(
                args.recon_root,
                dataset,
                args.frame,
                model,
                args.encoder,
                args.decoder,
                args.lpips_resolution,
                args.teacher_forcing,
            )
            for model in args.models
        ]
        rows.append(
            (
                dataset,
                find_raw_frame(args.raw_root, dataset, args.resolution, args.frame),
                find_masked_frame(args.mask_root, dataset, args.resolution, args.frame),
                recon_paths,
            )
        )

    column_count = 3 + len(args.models)
    width = (
        label_width
        + column_count * cell_width
        + (column_count + 1) * gap
        + colorbar_width
        + colorbar_label_width
    )
    height = header_height + len(rows) * cell_height + (len(rows) + 1) * gap
    grid = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(grid)

    for row_index, (dataset, raw_path, mask_path, recon_paths) in enumerate(rows):
        y = header_height + gap + row_index * (cell_height + gap)
        draw.rectangle((0, y, width, y + cell_height), fill=(255, 255, 255))
        label_x = (label_width - text_width(draw, dataset, font)) // 2
        label_y = y + (cell_height - text_height(draw, dataset, font)) // 2
        draw.text((label_x, label_y), dataset, fill=(25, 25, 25), font=font)

        images = [fit_image(raw_path, cell_width, cell_height), fit_image(mask_path, cell_width, cell_height)]
        images.extend(fit_image(path, cell_width, cell_height) for path in recon_paths)
        diff_image, diff_scale = difference_image(recon_paths[-2], recon_paths[-1], cell_width, cell_height)
        images.append(diff_image)

        for column, image in enumerate(images):
            x = label_width + gap + column * (cell_width + gap)
            grid.paste(image, (x, y))
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline=(210, 210, 210))

        colorbar_x = label_width + gap + len(images) * (cell_width + gap)
        draw_colorbar(draw, colorbar_x, y, colorbar_width, cell_height, diff_scale, colorbar_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)

    print(f"Saved {args.output}")
    print(f"Dimensions: {grid.size[0]} x {grid.size[1]}")
    print(f"Rows: {len(rows)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Encoder/decoder: {args.encoder} {args.decoder}")
    print(f"Frame: raw {frame4}, mask/recon {args.frame:06d}")


if __name__ == "__main__":
    main()
