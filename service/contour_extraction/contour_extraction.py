from PIL import Image

import cv2
import numpy as np

from pdf2image import convert_from_path


def load_image_from_path(path):
    # returns cv2 image
    preprocess_img = lambda img: cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    img = convert_from_path(path)[0]
    img = preprocess_img(img)

    return img


def load_image_from_pil(pil_img):
    # @param pil_img  image from pdf2img lib
    # return cv2 image
    preprocess_img = lambda img: cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    img = preprocess_img(pil_img)
    return img


def pil2cv(pil_img):
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)


def cv2pil(cv_img):
    return Image.fromarray(cv_img)


def show(cv_img):
    i = Image.fromarray(cv_img)
    return i.resize((i.size[0] // 2, i.size[1] // 2))


class RectsBank:
    # nms provider
    def __init__(self, intersec_lim=0.2):
        self.rects = []
        self.limit = intersec_lim

    def intersects(self, bbox):
        for bb in self.rects:
            if self._intersection(bb, bbox) > self.limit:
                return True
        return False

    def add(self, bbox):
        self.rects.append(bbox)

    def _intersection(self, b1, b2):
        xa1, ya1, wa, ha = b1
        xb1, yb1, wb, hb = b2
        xa2, ya2 = xa1 + wa, ya1 + ha
        xb2, yb2 = xb1 + wb, yb1 + hb

        # i stands for intersection
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)

        wi = xi2 - xi1
        hi = yi2 - yi1

        if wi <= 0 or hi <= 0:
            return 0.

        area = wi * hi
        minArea = min(wa * ha, wb * hb)
        return area / minArea


def get_white_area(thresh_img):
    # finding conturs
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find inner white area
    get_area = lambda c: -c[2] * c[3]
    contour_sizes = [(get_area(cv2.boundingRect(contour)), contour) for contour in contours]
    sorted_contours = sorted(contour_sizes, key=lambda x: x[0])

    mask = np.zeros(thresh_img.shape, dtype=np.uint8)
    img_area = thresh_img.shape[0] * thresh_img.shape[1]
    prev_contour = sorted_contours[0][1]
    for c in sorted_contours:
        if -c[0] / img_area < 0.5:
            crop_contour = prev_contour
            cv2.drawContours(mask, [crop_contour], -1, (255, 255, 255), cv2.FILLED)
            thresh_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
            thresh_img[mask == 0] = 255
            return thresh_img
        elif -cv2.contourArea(c[1]) / c[0] < 0.5:
            continue
        else:
            prev_contour = c[1]


def process_morph(img, kernel):
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    k = np.ones((kernel, kernel))
    img = cv2.dilate(img, kernel=k)
    img = cv2.erode(img, kernel=k)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return img


def find_conturs(cv_img):
    # return list of conturs bboxes in format [((x, y), (x1, y1)), ... ]

    # binarization
    thresh_img = cv2.threshold(cv_img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1]
    # make contours thicker
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(thresh_img, contours, -1, 0, 3)
    # get only inner area
    thresh_img = get_white_area(thresh_img)

    # finding contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [cv2.boundingRect(c) for c in contours]
    top_bboxes = sorted(bboxes, key=lambda b: -b[2] * b[3])

    rects_drawn = RectsBank()
    bboxes = []
    clim = 5
    c = 0
    b = 25  # border size in pixels
    for (x, y, w, h) in top_bboxes:
        if w * h / cv_img.size < 0.6 and w * h / cv_img.size > 0.03:
            # some kind of nms
            if rects_drawn.intersects((x, y, w, h)):
                # do not need submodules
                continue
            c += 1
            rects_drawn.add((x, y, w, h))
            bbox = ((x - b, y - b), (x + w + b, y + h + b))
            bboxes.append(bbox)

        if c >= clim:
            break
    return bboxes


def crop_conturs(cv_img):
    # returns list of detail images
    details = []
    bboxes = find_conturs(cv_img)
    for c, bbox in enumerate(bboxes):
        (x, y), (x1, y1) = bbox
        detail = cv_img[y:y1, x:x1]
        if detail.size > 0:
            details.append(detail)
    return details


def get_img_with_padding(image):
    COLOR = (255, 255, 255)
    (H, W) = image.shape[:2]
    pad_w = ((W + 31) // 32 * 32 - W + 32) // 2
    pad_h = ((H + 31) // 32 * 32 - H + 32) // 2
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, None, COLOR)

    return image

def get_contour(img, kernel=7):
    # getting out if found no contours while iterations
    if kernel == 1:
        return None

    # preprocess images
    img_re = process_morph(img, kernel)

    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_re, contours[1:], -1, 0, 5)
    k = np.ones((55, 55))
    k_d = np.ones((50, 50))

    img_re = cv2.erode(img_re, kernel=k)
    img_re = cv2.dilate(img_re, kernel=k_d)

    cv2.rectangle(img_re, (0, 0), (img_re.shape[1], img_re.shape[0]), (255, 255, 255), 5)
    # find closed contours
    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get out if we found nothing
    if len(contours) < 2:
        return get_contour(img, kernel - 1)

    # get the biggest contour
    get_area = lambda c: -c[2] * c[3]
    contour_sizes = [(get_area(cv2.boundingRect(contour)), contour) for contour in contours]
    b_contour = sorted(contour_sizes, key=lambda x: x[0])[1]

    if (-b_contour[0] / img.size > 0.15) or (
            -b_contour[0] / img.size > 0.08 and ((-b_contour[0] - cv2.contourArea(b_contour[1])) / img.size < 0.1)):
        return b_contour[1]
    else:
        return get_contour(img, kernel - 1)