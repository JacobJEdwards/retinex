import logging
from pathlib import Path

import tqdm

import retinex
import cv2
import numpy as np
from tqdm.contrib.concurrent import process_map
from typing import Final, Sequence, Iterable

RETINEX_SIGMA_LIST: Final[Sequence[int]] = [15, 80, 250]

logger = logging.getLogger(__name__)


def global_luminance_stats(
    imgs: Iterable[np.ndarray],
) -> tuple[np.floating, np.floating]:
    all_l = []
    for img in imgs:
        lum, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        all_l.append(lum.flatten())
    all_l = np.concatenate(all_l)
    return np.mean(all_l), np.std(all_l)


def match_luminance(
    img: np.ndarray, global_mean: np.floating, global_std: np.floating
) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    local_mean, local_std = np.mean(lum), np.std(lum)
    if local_std > 0:
        lum = (lum - local_mean) * (global_std / local_std) + global_mean
    lum = np.clip(lum, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)


def process_image(image: np.ndarray) -> np.ndarray:
    processed_image = retinex.MSRCP(
        image, sigma_list=RETINEX_SIGMA_LIST, low_clip=0.01, high_clip=0.99
    )

    tqdm.tqdm.write("Processing image...")

    return processed_image


def process_and_save(img: tuple[Path, np.ndarray]) -> tuple[Path, np.ndarray]:
    path, image = img

    tqdm.tqdm.write(f"Processing image {path.name}")

    processed = process_image(image)

    cv2.imwrite(str(path), processed)

    return path, processed


def run(input_dir: Path, output_dir: Path) -> None:
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))

    imgs: list[tuple[Path, np.ndarray]] = [
        (output_dir / "retinex" / image_path.name, cv2.imread(str(image_path)))
        for image_path in image_paths
    ]

    processed_imgs = process_map(process_and_save, imgs, max_workers=4, chunksize=1)

    global_mean, global_std = global_luminance_stats(img for _, img in processed_imgs)

    logger.info(f"Global mean: {global_mean}")
    logger.info(f"Global std: {global_std}")

    logger.info(f"Processing images: {len(processed_imgs)}")

    for output_path, processed_image in tqdm.tqdm(processed_imgs):
        final_image = match_luminance(processed_image, global_mean, global_std)

        tqdm.tqdm.write(f"Writing {output_path.name}")

        p = output_dir / "final" / output_path.name

        cv2.imwrite(str(p), final_image)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Retinex Image Processing")
    parser.add_argument("input_dir", type=str, help="Path to the input images")
    parser.add_argument(
        "output_dir", type=str, help="Path to save the processed images"
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    assert input_dir.is_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retinex").mkdir(parents=True, exist_ok=True)
    (output_dir / "final").mkdir(parents=True, exist_ok=True)

    run(input_dir, output_dir)


if __name__ == "__main__":
    main()
