from typing import List, Any

import cv2
import numpy as np
from tqdm import tqdm

from last_five import LastFiveRatio


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 22)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((4, 4), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    return image


class QRPatternFinder:
    def __init__(self, image):
        self.image = image
        self.ratio = None
        self.row = 0
        self.col = 0

    def check_column(self):
        center_x = self.ratio.get_center()
        seg_len = self.ratio.segment_len()

        check_col = False
        for center in range(center_x - int(seg_len * 0.3), center_x + int(seg_len * 0.3)):
            col_ratio = LastFiveRatio(first_three_black=True)
            column = self.image[self.row - seg_len // 2: self.row + seg_len // 2, center]
            for pixel in column:
                col_ratio.add_pixel(pixel, None)

            if col_ratio.is_pattern():
                check_col = True
                break
        return check_col

    def get_rectangle(self):
        center_y = self.row
        center_x = self.ratio.get_center()
        seg_len = self.ratio.segment_len()

        return self.image[
               center_y - seg_len // 2: center_y + seg_len // 2,
               center_x - seg_len // 2: center_x + seg_len // 2
               ]

    def check_main_diagonal(self):
        main_diag = np.diag(self.get_rectangle())
        main_diag_ratio = LastFiveRatio(first_three_black=True)
        for pixel in main_diag:
            main_diag_ratio.add_pixel(pixel, None)

        return main_diag_ratio.is_pattern()

    def check_second_diagonal(self):
        second_diag = np.diag(np.fliplr(self.get_rectangle()))
        second_diag_ratio = LastFiveRatio(first_three_black=True)
        for pixel in second_diag:
            second_diag_ratio.add_pixel(pixel, None)

        return second_diag_ratio.is_pattern()

    def check_row(self):
        self.ratio = LastFiveRatio()
        coordinates = []
        for col in range(self.image.shape[1]):
            self.col = col

            # Step 1 - does row satisfy the pattern
            self.ratio.add_pixel(self.image[self.row][col], col)
            if not self.ratio.is_pattern():
                continue

            # Step 2 - does column satisfy the pattern
            if not self.check_column():
                continue

            # Step 3 - does main diagonal satisfy the pattern
            if not self.check_main_diagonal():
                continue

            # Step 4 - does second diagonal satisfy the pattern
            if not self.check_second_diagonal():
                continue

            # Save the coordinates of the found rectangle
            center_y = self.row
            center_x = self.ratio.get_center()
            coordinates.append(np.array([center_x, center_y]))

        return coordinates

    def find_qr_patterns(self) -> np.ndarray:
        self.image = preprocess_image(self.image)

        coordinates = []
        for row in tqdm(range(0, self.image.shape[0], 3), position=1, leave=False, desc="QR Recognition"):
            self.row = row
            coordinates += self.check_row()

        return np.array(filter_close_points(coordinates))


def filter_close_points(coordinates: List[Any]) -> List[Any]:
    EPS = 20
    new_coords = []
    for coord in coordinates:
        is_good = True
        for new_coord in new_coords:
            if np.sqrt(np.sum((coord - new_coord) ** 2)) < EPS:
                is_good = False
                break

        if is_good:
            new_coords.append(coord)

    return new_coords
