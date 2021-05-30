from dataclasses import dataclass
import numpy as np
import cv2

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

def get_bbox(contour):
    bbox = cv2.boundingRect(contour)
    x0,y0,w,h = bbox
    return Bbox(x0, y0, x0+w, y0 + h)

def get_subimg(img, bbox:Bbox):
    return img[bbox.y0:bbox.y1, bbox.x0:bbox.x1]

def combine_bboxes(mbbox, rbbox):
    # mbbox in main bbox
    # rbbox is relative to mbbox
    x0, y0 = mbbox.x0, mbbox.y0
    return Bbox(x0+rbbox.x0, y0+rbbox.y0, x0+rbbox.x1, y0+rbbox.y1)