import cv2
import numpy as np
from . import utils as u
import math
from .bbox import Bbox, get_subimg, combine_bboxes
from .line import Line, create_line_from_HoughP


class ImageConstants:
    def __init__(self, img):
        self.img = img
        self.H, self.W = img.shape
        self.SAME_LINES_DIFF = int(self.H * 0.005) + 1
        self.BORDER_EPSILON = int(self.H*0.05)
        self.BIG_VERTICAL_GAP = int(self.H * 0.09)
        
            
    def almost_0(self, value): return np.abs(value) < self.SAME_LINES_DIFF
    def equivalent(self, value1, value2): return self.almost_0(value1-value2)


def remove_text(img):
    return u.extract_contours(img)

def get_lower_image(contour_img, consts:ImageConstants)->Bbox:
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

def _get_lines(img):
    minLineLength = 100
    maxLineGap = 10

    #  Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 200, None, minLineLength, maxLineGap)
    lines = [create_line_from_HoughP(l) for l in lines]

    # select vlines & hlines
    hlines = sorted(filter(lambda x: x.is_horizontal(), lines), key=lambda x:x.mean_y())
    vlines = sorted(filter(lambda x: x.is_vertical(), lines), key=lambda x:x.mean_x())
    return hlines, vlines

def _get_table_old(contour_lower_img, consts)->Bbox:
    t_vh = contour_lower_img.copy()
    hlines, vlines = _get_lines(t_vh)

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

def _get_first_approx_bbox(lower_img, consts):
    '''Cuts lower right corner of  wide drawings'''
    
    t_vh = lower_img.copy()
    H,W = t_vh.shape
    hlines, vlines = _get_lines(t_vh)
    
    # find lower and left bounds of table
    bottom_y = max([max(l.y1, l.y2) for l in hlines])
    left_x = W - int(3.3 * bottom_y)
    
    # return bbox
    eps = consts.SAME_LINES_DIFF
    return Bbox(max(0,left_x-eps), 0, W, min(H,bottom_y+eps))

def get_table(img, consts)->Bbox:
    t_vh = img.copy()
    # get lower image
    lower_img_bbox = get_lower_image(t_vh, consts)
    lower_img = get_subimg(t_vh, lower_img_bbox)
    # get table image
    first_approx_bbox = _get_first_approx_bbox(lower_img, consts)
    first_approx_img = get_subimg(lower_img, first_approx_bbox)
    table_bbox = _get_table_old(first_approx_img, consts)
    combined_bbox = combine_bboxes(mbbox=first_approx_bbox, rbbox=table_bbox)
    # combine lower with table
    return combine_bboxes(mbbox=lower_img_bbox, rbbox=combined_bbox)
