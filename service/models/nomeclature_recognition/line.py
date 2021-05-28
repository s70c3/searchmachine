import cv2
import numpy as np
from . import utils as u
from dataclasses import dataclass


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    def mean_y(self): return (self.y1 + self.y2) / 2
    def mean_x(self): return (self.x1 + self.x2) / 2
    def min_x(self): return min(self.x1, self.x2)
    def max_x(self): return max(self.x1, self.x2)
    def min_y(self): return min(self.y1, self.y2)
    def max_y(self): return max(self.y1, self.y2)
    def _spread_y(self): return np.abs(self.y2 - self.y1)
    def _spread_x(self): return np.abs(self.x2 - self.x1)
    def is_horizontal(self): return self._spread_y() < self._spread_x() / 2
    def is_vertical(self): return self._spread_x() < self._spread_y() / 2
    def __str__(self): return f"({self.x1}, {self.y1}), ({self.x2}, {self.y2})"
    def __repr__(self): return str(self)


def create_line_from_HoughP(line):
    x1, y1, x2, y2 = line[0]
    return Line(x1, y1, x2, y2)


def show_lines(img, lines, window_mode=cv2.WINDOW_NORMAL):
    i = 0
    while True:
        line = lines[i]
        x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
        cdst = u.gray2rgb(img.copy())
        cdst = cv2.line(cdst, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.namedWindow(str(i), window_mode)
        cv2.imshow(str(i), cdst)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 81:
            if i > 0: i -= 1
            continue
        elif key == 83:
            if i < len(lines) - 1: i += 1
            continue
        else:
            break