from typing import Sequence

import numpy as np
import cv2


def multi_scale_retinex(img: np.ndarray, sigma_list: Sequence[float]) -> np.ndarray:
    img_log = np.log10(img)
    blur_logs = np.empty((len(sigma_list), *img.shape), dtype=np.float64)

    for i, sigma in enumerate(sigma_list):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        blur_logs[i] = img_log - np.log10(blur)

    retinex = np.mean(blur_logs, axis=0)
    return retinex


def color_restoration(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    img_sum = np.sum(img, axis=2, keepdims=True)

    cr = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return cr


def simplest_color_balance(
    img: np.ndarray, low_clip: float, high_clip: float
) -> np.ndarray:
    low_percentile = low_clip * 100
    high_percentile = high_clip * 100

    for i in range(img.shape[2]):
        channel = img[:, :, i]
        low_val = np.percentile(channel, low_percentile, method="lower")
        high_val = np.percentile(channel, high_percentile, method="lower")
        img[:, :, i] = np.clip(channel, low_val, high_val)

    return img


def MSRCR(
    img: np.ndarray,
    sigma_list: Sequence[float],
    G: float,
    b: float,
    alpha: float,
    beta: float,
    low_clip: float,
    high_clip: float,
) -> np.ndarray:
    img = img.astype(np.float64) + 1.0

    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    min_val = img_msrcr.min(axis=(0, 1), keepdims=True)
    max_val = img_msrcr.max(axis=(0, 1), keepdims=True)

    denominator = np.where(max_val - min_val == 0, 1, max_val - min_val)

    img_msrcr = (img_msrcr - min_val) / denominator * 255.0
    img_msrcr = np.clip(img_msrcr, 0, 255)
    img_msrcr = np.uint8(img_msrcr)
    img_msrcr = simplest_color_balance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img: np.ndarray, sigma_list: Sequence[float]) -> np.ndarray:
    img = img.astype(np.float32) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        channel = img_retinex[:, :, i]

        hist_data = (channel * 100).astype(np.int32)
        unique, counts = np.unique(hist_data, return_counts=True)

        zero_idx = np.where(unique == 0)[0]
        zero_count = counts[zero_idx[0]] if len(zero_idx) > 0 else 0
        threshold = zero_count * 0.1

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        for u, c in zip(unique, counts):
            if u < 0 and c < threshold:
                low_val = u / 100.0
            if u > 0 and c < threshold:
                high_val = u / 100.0
                break

        channel = np.clip(channel, low_val, high_val)
        channel_min = np.min(channel)
        channel_range = np.max(channel) - channel_min
        img_retinex[:, :, i] = (channel - channel_min) / channel_range * 255

    return img_retinex.astype(np.uint8)


def MSRCP(
    img: np.ndarray, sigma_list: Sequence[float], low_clip: float, high_clip: float
) -> np.ndarray:
    img = img.astype(np.float64) + 1.0

    intensity = np.mean(img, axis=2, keepdims=True)

    retinex = multi_scale_retinex(intensity[:, :, 0], sigma_list)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplest_color_balance(retinex, low_clip, high_clip)

    intensity1_min = np.min(intensity1)
    intensity1_max = np.max(intensity1)

    intensity1 = (intensity1 - intensity1_min) / (
        intensity1_max - intensity1_min
    ) * 255.0 + 1.0

    B = np.max(img, axis=2, keepdims=True)
    A = np.minimum(256.0 / B, intensity1 / intensity)

    img_msrcp = A * img
    img_msrcp = np.clip(img_msrcp - 1.0, 0, 255)
    img_msrcp = img_msrcp.astype(np.uint8)

    return img_msrcp
