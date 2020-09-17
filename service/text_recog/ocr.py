from PIL import Image
import numpy as np

import cv2
import pyocr
import pyocr.builders

from .detection import crop_conturs, pil2cv

tool = pyocr.get_available_tools()[0]
langs = tool.get_available_languages()

print("Will use tool '%s'" % (tool.get_name()))
print("Available languages: %s" % ", ".join(langs))


__all__ = ['extract_sizes']


def convert(word_box):
    (minx, miny), (maxx, maxy) = word_box.position
    poly_np = np.array([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
    ])
    return poly_np, word_box.content


# rn
# poly_np, word_box.content


class MyBuilder(pyocr.builders.WordBoxBuilder):
    def __init__(self):
        super().__init__()
        self.tesseract_configs = ["-c", "tessedit_char_whitelist=0123456789+*,.=x±"] + self.tesseract_configs


def ocr(img, psm=11):
    builder = MyBuilder()
    #     builder = pyocr.builders.WordBoxBuilder()
    #     builder = pyocr.builders.DigitBuilder()
    # builder.tesseract_flags = ['--psm','1']
    builder.tesseract_flags = ['--psm', str(psm)]
    word_boxes = tool.image_to_string(
        img,
        lang="eng",
        builder=builder
    )
    polys_np, labels = zip(*[convert(word_box) for word_box in word_boxes]) if word_boxes else ([], [])
    return polys_np, labels


def rotated_ocr(img, angle, builder=pyocr.builders.WordBoxBuilder()):
    img = img.rotate(angle, expand=True)
    polys_np, labels = ocr(img)
    return labels


DILATION_SIZE = 2
EROSION_SIZE = 3


def dilate(cv_img, kernel=2):
    return cv2.dilate(cv_img, np.ones((kernel, kernel), np.uint8))


def erode(cv_img, kernel=2):
    return cv2.erode(cv_img, np.ones((kernel, kernel), np.uint8))


def show(path):
    img = cv2.imread(path)
    return Image.fromarray(img)


def recorgnize(img):
    polys_np, rec_scores = [], []
    labels = {-90: [], 0: []}

    for i, angle in enumerate([0, -90]):
        labels0 = rotated_ocr(img, angle)  # , builder=builder)
        labels[angle].extend(labels0)
    return labels


def process(pil_img):
    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    #     img = erode(dilate(img, DILATION_SIZE), EROSION_SIZE)
    img = Image.fromarray(img)
    labels = recorgnize(img)
    return labels


def filt(variants):
    return list(filter(lambda v: v != ' ' and any(list(map(str.isdigit, v))), variants))


LINEAR_MIN = 10
LINEAR_MAX = 5000


def extract_sizes(pil_img):
    def get_linear_size(w):
        if len(w) == 0:
            return None

        if not any(list(map(str.isdigit, w))):
            return None

        sizes = []
        start_i = -1
        for i in range(len(w)):
            if w[i].isdigit():
                # find digit
                if start_i == -1:
                    start_i = i
                if i == len(w) - 1:
                    size = w[start_i:len(w)]
                    sizes.append(size)
            else:
                # digit ends
                if start_i != -1:
                    size = w[start_i:i]
                    sizes.append(size)
                start_i = -1
        if len(sizes) == 0:
            return None

        # delete probably papers ids
        if '4447' in sizes:
            sizes.remove('4447')
        if '4465' in sizes:
            sizes.remove('4465')

        sizes = list(map(int, sizes))
        for size in sizes:
            if size > 9999 and size % 100 // 10 == 4:
                size = size // 100
        maxsize = max(sizes)
        if LINEAR_MIN < maxsize < LINEAR_MAX:
            return maxsize

    recognized = {-90: [], 0: []}
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    for pic in crop_conturs(cv_img):
        contours, hierarchy = cv2.findContours(pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(pic, contours, -1, 0, 1)
        projection_recognized = process(pic)
        for angled in projection_recognized:

            recognized[angled] += filt(projection_recognized[angled])
    recog_linears = {}
    for angled in recognized:
        recog_linears[angled] = list(map(get_linear_size, recognized[angled]))
        recog_linears[angled] = list(filter(lambda v: v is not None, recog_linears[angled]))
    linears = []
    all_sizes = []
    for angled in recog_linears:
        all_sizes.extend(recog_linears[angled])
        if len(recog_linears[angled]) > 0:
            linears.append(max(recog_linears[angled]))
    # return a tuple with 2 sizes (height and width) and all sizes
    return linears, all_sizes
