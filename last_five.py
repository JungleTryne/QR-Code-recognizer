import numpy as np

BLACK = 0
WHITE = 255


class LastFiveRatio:
    """
    Responsible for storing a buffer with last
    B-W-B-W-B pixels. Helps determine if QR pattern is found
    """

    def __init__(self, first_three_black=False):
        self.counters = []
        self.coordinates = []
        self.last_color = None
        self.first_three_black = first_three_black
        self.black_counter = 0

    def add_pixel(self, pixel_color, x):
        if pixel_color == WHITE and self.first_three_black and self.black_counter >= 3:
            return

        if self.last_color == pixel_color:
            self.counters[-1] += 1
            return

        if self.last_color is None:
            if pixel_color == BLACK:
                self.black_counter += 1
                self.counters.append(1)
                self.coordinates.append(x)
                self.last_color = BLACK
            return

        self.counters.append(1)
        self.coordinates.append(x)
        self.last_color = pixel_color

        if len(self.counters) > 5:
            self.counters = self.counters[1:]
            self.coordinates = self.coordinates[1:]

        if self.last_color == BLACK:
            self.black_counter += 1

    def get_center(self):
        if len(self.counters) < 5:
            return None
        return self.coordinates[0] + self.segment_len() // 2

    def is_pattern(self):
        if len(self.counters) < 5:
            return 0

        if self.last_color == WHITE:
            return 0

        ratio = np.array(self.counters, dtype=np.float64)
        ratio /= ratio[0]

        targets = [
            np.array([1.0, 1.0, 3.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 6.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 6.0, 2.0, 2.0])
        ]

        result = False
        for target in targets:
            result = result or np.isclose(ratio, target, rtol=0.4).all()

        return result

    def segment_len(self):
        return np.sum(np.array(self.counters))
