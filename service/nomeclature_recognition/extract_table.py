import cv2
import numpy as np
import nomeclature_recognition.utils as u
import math
from dataclasses import dataclass

@dataclass
class Line:
    x1:int
    y1:int
    x2:int
    y2:int
        
    def mean_y(self): return (self.y1 + self.y2) / 2
    def mean_x(self): return (self.x1 + self.x2) / 2
    def min_x(self): return min(self.x1, self.x2)
    def max_x(self): return max(self.x1, self.x2)
    def min_y(self): return min(self.y1, self.y2)
    def max_y(self): return max(self.y1, self.y2)
    def _spread_y(self): return np.abs(self.y2-self.y1)
    def _spread_x(self): return np.abs(self.x2-self.x1)
    def is_horizontal(self): return self._spread_y() < self._spread_x()/2
    def is_vertical(self): return self._spread_x() < self._spread_y()/2
    def __str__(self): return f"({self.x1}, {self.y1}), ({self.x2}, {self.y2})"
    def __repr__(self): return str(self)
    
def create_line_from_HoughP(line):
    x1, y1, x2, y2 = line[0]
    return Line(x1, y1, x2, y2)
    
    
def show_lines(img, lines, window_mode=cv2.WINDOW_NORMAL):
    i = 0
    while True:
        line = lines[i]
        x1,y1,x2,y2 = line.x1,line.y1,line.x2,line.y2
        cdst = u.gray2rgb(img.copy())
        cdst = cv2.line(cdst,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.namedWindow(str(i), window_mode)
        cv2.imshow(str(i), cdst)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 81:
            if i > 0: i-= 1
            continue
        elif key == 83:
            if i < len(lines)-1: i += 1
            continue
        else:
            break
            
class ImageConstants:
    def __init__(self, img):
        self.img = img
        self.H, self.W = img.shape
        self.SAME_LINES_DIFF = int(self.H * 0.005) + 1
        self.BORDER_EPSILON = int(self.H*0.05)
        self.BIG_VERTICAL_GAP = int(self.H * 0.09)
        
            
    def almost_0(self, value): return np.abs(value) < self.SAME_LINES_DIFF
    def equivalent(self, value1, value2): return self.almost_0(value1-value2)


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


class CellExtractor:
    def __init__(self, table, consts):
        self.table = table
        self.consts = consts
        
    def _almost_0(self, v): return self.consts.almost_0(v)
    def _equiv(self, v1, v2): return self.consts.equivalent(v1, v2)
        
    def _get_cells(self):
        contours, hierarchy = cv2.findContours(self.table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = list(map(get_bbox, contours))
        # filter out too large and lines
        H, W = self.table.shape
        table_area = H * W
        is_too_large = lambda x: x.area() > table_area / 2
        is_a_line = lambda x: self._almost_0(x.width()) or self._almost_0(x.height())
        is_a_cell = lambda x: not is_too_large(x) and not is_a_line(x)
        bboxes = list(filter(is_a_cell, bboxes))
        return bboxes
    
    def _get_left_neighbour(self, cell, bboxes):
        x0, y1 = cell.x0, cell.y1
        criterion = lambda x: self._equiv(x.x1, x0) and self._equiv(x.y1, y1)
        candidates = list(filter(criterion, bboxes))
        if len(candidates) == 1: return candidates[0]
        else: return None
    
    def get_materical_cell(self):
        bboxes = self._get_cells()
        
        # get lowest cells
        max_low = max([bbox.y1 for bbox in bboxes])
        lowest_cells = list(filter(lambda x: self._equiv(x.y1, max_low), bboxes))

        lowest_cells_left_to_right = sorted(lowest_cells, key=lambda x: x.x1)
        material_bbox = lowest_cells_left_to_right[-2]
        return material_bbox
    
    def _get_rightest_cell(self, nrow, bboxes):
        # counting from bottom, starting at 0
        max_right = max([bbox.x1 for bbox in bboxes])
        rightest_cells = list(filter(lambda x: self._equiv(x.x1, max_right), bboxes))
        rightest_cells_top_to_bottom = sorted(rightest_cells, key=lambda x: x.y1)
        return rightest_cells_top_to_bottom[-(nrow+1)]
    
    def get_mass_cell(self):
        bboxes = self._get_cells()
        second_row_rightest_cell = self._get_rightest_cell(2, bboxes)
        return self._get_left_neighbour(second_row_rightest_cell, bboxes)
    
    def get_mass_header_cell(self):
        bboxes = self._get_cells()
        third_row_rightest_cell = self._get_rightest_cell(3, bboxes)
        return self._get_left_neighbour(third_row_rightest_cell, bboxes)
    
def remove_text(img):
    return u.extract_contours(img)

def get_lower_image(contour_img, consts:ImageConstants)->Bbox:
    H, W = contour_img.shape
    t_vh = contour_img.copy()
    
    #  Standard Hough Line Transform
    lines = cv2.HoughLines(t_vh, 1, np.pi / 180, 400, None, 0, 0)

    def mykey(x): return x[0][1], x[0][0]
    lines = sorted(lines, key=mykey)

    horizontal_lines = [l for l in lines if np.abs(l[0][1] - math.pi/2) < 1e-4]

    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[0][0], reverse=True)

    lower_lines = [horizontal_lines[0]]

    H,W = t_vh.shape
    for l in horizontal_lines[1:]:
        rho_old = lower_lines[-1][0][0]
        rho_cur = l[0][0]
        distance = rho_old - rho_cur
        relative_distance = distance / H
        if distance < consts.SAME_LINES_DIFF:
            # these are the same actual line
            continue
        elif distance > consts.BIG_VERTICAL_GAP:
            # this is the large gap after the table
            break
        else:
            # these are different lines of the table (or lower)
            lower_lines.append(l)

    return Bbox(0, int(lower_lines[-1][0][0] - consts.BORDER_EPSILON), W, H)

def get_table(contour_lower_img, consts)->Bbox:
    t_vh = contour_lower_img.copy()
    minLineLength = 100
    maxLineGap = 10

    #  Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(t_vh, 1, np.pi / 180, 200, None, minLineLength, maxLineGap)
    lines = [create_line_from_HoughP(l) for l in lines]

    hlines = list(filter(lambda x: x.is_horizontal(), lines))
    vlines = list(filter(lambda x: x.is_vertical(), lines))

    hlines = sorted(hlines, key=lambda x:x.mean_y())
    vlines = sorted(vlines, key=lambda x:x.mean_x())

    top_line = hlines[0]
    top_lines = [l for l in hlines if np.abs(l.mean_y() - top_line.mean_y()) < consts.SAME_LINES_DIFF]

    min_left_x = min([l.min_x() for l in top_lines])
    max_right_x = max([l.max_x() for l in top_lines])
    left_borders = [l for l in vlines if np.abs(l.mean_x() - min_left_x) < consts.SAME_LINES_DIFF]
    right_borders = [l for l in vlines if np.abs(l.mean_x() - max_right_x) < consts.SAME_LINES_DIFF]

    max_down_left = max(map(lambda x: x.max_y(), left_borders))
    max_down_right = max(map(lambda x: x.max_y(), right_borders))
    max_down = max(max_down_left, max_down_right)
    min_up = min([l.min_y() for l in top_lines])

    eps = consts.SAME_LINES_DIFF
    table_bbox = Bbox(min_left_x-eps, min_up-eps, max_right_x+eps, max_down+eps)
    return table_bbox
