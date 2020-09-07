from dataclasses import dataclass
import numpy as np


@dataclass
class Bbox:
    x0: int
    y0: int
    x1: int
    y1: int

    def to_contour(self):
        return np.array([
            [[self.x0, self.y0]],
            [[self.x1, self.y0]],
            [[self.x1, self.y1]],
            [[self.x0, self.y1]]
        ])

    def height(self): return self.y1 - self.y0
    def width(self): return self.x1 - self.x0
    def area(self): return self.height() * self.width()


@dataclass
class TextBbox:
    bbox: Bbox
    text: str

    def __str__(self): return f'{self.bbox}: text=[{self.text}]'
    def __repr__(self): return str(self)