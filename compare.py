from typing import Tuple

import click
import os

import cv2
import numpy as np
from tqdm import tqdm

from cv_qr_finder import CVQRPatternFinder
from qr_finder import QRPatternFinder


def get_metrics(result: np.ndarray, target: np.ndarray) -> Tuple[int, int, int]:
    EPS = 150

    true_pos_result = np.zeros(len(result))
    true_pos_target = np.zeros(len(target))

    for i, left in enumerate(result):
        for j, right in enumerate(target):
            if np.sqrt(np.sum((left - right) ** 2)) < EPS:
                true_pos_result[i] = 1
                true_pos_target[j] = 1

    TP = int(np.sum(true_pos_target))
    FN = len(target) - np.sum(true_pos_target)
    FP = len(result) - np.sum(true_pos_result)

    return TP, FN, FP


@click.command()
@click.option("--images-path", type=click.Path(exists=True), required=True, help="Path to image with QR-Codes")
def main(images_path: click.Path):
    images = os.listdir(str(images_path))

    TP, FN, FP = 0, 0, 0
    for image_name in tqdm(images, position=0, desc="Benchmark"):
        image_tensor = cv2.imread(os.path.join(str(images_path), image_name), cv2.IMREAD_GRAYSCALE)

        pattern_finder_own = QRPatternFinder(image_tensor)
        pattern_finder_cv2 = CVQRPatternFinder(image_tensor)

        own_points = pattern_finder_own.find_qr_patterns()
        cv2_points = pattern_finder_cv2.find_qr_patterns()

        if cv2_points is None:
            print(f"failed to assess {image_name}")
            continue

        TP_new, FN_new, FP_new = get_metrics(own_points, cv2_points)
        TP += TP_new
        FN += FN_new
        FP += FP_new

    print(f"Precision: {TP / (TP + FP)}, Recall: {TP / (TP + FN)}")


if __name__ == "__main__":
    main()