import retinex
import cv2
import numpy as np
from typing import Final, Sequence

RETINEX_SIGMA_LIST: Final[Sequence[int]] = [15, 80, 250]


def main() -> None:
    random_image = np.random.random_sample((100, 100, 3)) * 255

    adjusted = retinex.MSSRP(
        random_image, RETINEX_SIGMA_LIST, low_clip=0.01, high_clip=0.99
    )

    cv2.imshow("Example", adjusted)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
