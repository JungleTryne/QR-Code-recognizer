from typing import Optional

import cv2
import numpy as np


class CVQRPatternFinder:
    """
    QR code finder using cv2 library
    """
    def __init__(self, image: np.ndarray):
        self.image = image

    def find_qr_patterns(self) -> Optional[np.ndarray]:
        detector = cv2.QRCodeDetector()

        _, _, points, _ = detector.detectAndDecodeMulti(self.image)
        if points is None:
            return None

        points = np.concatenate(points)
        new_points = [point for i, point in enumerate(points) if i % 4 != 2]

        return np.array(new_points)