import ezdxf
import cv2
import numpy as np

def preprocess_dxf_file(dxf_name):
    dxf = ezdxf.readfile(dxf_name)
    msp = dxf.modelspace()
    mask = np.zeros((10000, 10000), dtype="uint8")
    # shift for centering
    c = 5000
    for l in msp:
        if l.dxftype() == "LINE":
            x1, y1 = list(map(round, l.dxf.start[0:2]))
            x2, y2 = list(map(round, l.dxf.end[0:2]))
            cv2.line(mask, (x1 + c, y1 + c), (x2 + c, y2 + c), 255, 1)
        if l.dxftype() == "CIRCLE":
            x1, y1 = list(map(round, l.dxf.center[0:2]))
            cv2.circle(mask, (x1 + c, y1 + c), round(l.dxf.radius), 255, thickness=1, lineType=8, shift=0)
        if l.dxftype() == "LWPOLYLINE":
            points = l.get_points("xyb")
            prev_point = points[0]
            for i in range(1, len(points)):
                point = points[i]
                if abs(prev_point[2]) == 1:
                    center_x = (round(prev_point[0]) + round(point[0])) // 2
                    center_y = (round(prev_point[1]) + round(point[1])) // 2
                    major_axe = abs(round(point[0]) - round(prev_point[0])) // 2
                    minor_axe = abs(round((round(point[0]) - round(prev_point[0]))) // 2)
                    cv2.ellipse(mask, (center_x + c, center_y + c), (major_axe, minor_axe), 0.0, 0.0, 360,
                                (255, 255, 255), 1)
                else:
                    x1, y1 = list(map(round, prev_point[0:2]))
                    x2, y2 = list(map(round, point[0:2]))
                    cv2.line(mask, (x1 + c, y1 + c), (x2 + c, y2 + c), 255, 1)
                prev_point = point
            if l.closed:
                x1, y1 = list(map(round, points[len(points) - 1][0:2]))
                x2, y2 = list(map(round, points[0][0:2]))
                cv2.line(mask, (x1 + c, y1 + c), (x2 + c, y2 + c), 255, 1)
        if l.dxftype() == "ARC":
            x1, y1 = list(map(round, l.dxf.center[0:2]))
            cv2.ellipse(mask, (x1 + c, y1 + c), (round(l.dxf.radius), round(l.dxf.radius)), 0, round(l.dxf.start_angle),
                        round(l.dxf.end_angle), 255, thickness=1, lineType=8, shift=0)
    # returning an image
    return mask


def get_min(cont):
    minX = np.min(cont[:, 0, 0])
    minY = np.min(cont[:, 0, 1])
    return minX, minY


def get_outer_rect(image):
    return get_outer_contour_with_info(image)[1]


def get_outer_contour(image):
    return get_outer_contour_with_info(image)[2]


def get_outer_contour_with_info(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # find the outer contour
    get_area = lambda c: -c[2] * c[3]
    contour_sizes = [(get_area(cv2.boundingRect(contour)), cv2.boundingRect(contour), contour) for contour in contours]
    biggest_contour = sorted(contour_sizes, key=lambda x: x[0])[0]
    return biggest_contour


def extract_info(dxf_name):
    thresh_img = preprocess_dxf_file(dxf_name)
    x1, y1, w, h = get_outer_rect(thresh_img)
    # normalize coordinates
    contour = get_outer_contour(thresh_img)
    minX, minY = get_min(contour)
    cnt_norm = contour - [minX, minY]
    return cnt_norm, (w, h)


def get_contour_from_dxf(dxf_name):
    """
    Main method that extracts contour from dxf as a np.ndarray
    @param dxf_name: path to dxf file
    @return: contour np.ndarray
    """
    contour = extract_info(dxf_name)[0]
    contour = contour.squeeze()
    return contour


def get_size_from_dxf(dxf_name):
    return extract_info(dxf_name)[1]
