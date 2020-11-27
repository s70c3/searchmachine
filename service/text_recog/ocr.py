from PIL import Image
import numpy as np

import cv2
import pyocr.builders

from .detection import crop_conturs, pil2cv, cv2pil

from imutils.object_detection import non_max_suppression

tool = pyocr.get_available_tools()[0]
langs = tool.get_available_languages()

tool = pyocr.get_available_tools()[0]
langs = tool.get_available_languages()

print("Will use tool '%s'" % (tool.get_name()))
print("Available languages: %s" % ", ".join(langs))

__all__ = ['extract_sizes']


def get_img_with_padding(image):
    COLOR = (255, 255, 255)
    (H, W) = image.shape[:2]
    pad_w = ((W + 31) // 32 * 32 - W) // 2
    pad_h = ((H + 31) // 32 * 32 - H) // 2
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, None, COLOR)

    return image


def get_text_boxes(img):
    def dilate_ft(cv_img, kernel=30):
        return cv2.dilate(cv_img, np.ones((kernel, kernel), np.uint8))

    def erode_ft(cv_img, kernel=20):
        return cv2.erode(cv_img, np.ones((kernel, kernel), np.uint8))

    cleaned = hough_cleaning(img.copy())
    mask = dilate_ft(erode_ft(cleaned))
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    #     mask = cv2.bitwise_not(mask)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return_mask = np.ones(img.shape) * 255

    if len(contours) > 1:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            return_mask[y - 30:y + h + 30, x - 30:x + w + 30] = cleaned[y - 30:y + h + 30, x - 30:x + w + 30]

    return return_mask

def process_for_getting_only_text(img):
    img = get_img_with_padding(img)
    mask = get_text_boxes(img)
    return mask

def hough_cleaning(img):
    #      threshhold, threshhold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img, 150, 200, 3, 5)
    #     show_one(edges)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi, threshold=100, minLineLength=100, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    # Horizontal lines

    lines2 = cv2.HoughLinesP(
        edges, 1, np.pi / 2, threshold=100, minLineLength=100, maxLineGap=5)
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return img


def convert(word_box):
    (minx, miny), (maxx, maxy) = word_box.position
    poly_np = np.array([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
    ])
    return poly_np, word_box.content


class MyBuilder(pyocr.builders.WordBoxBuilder):
    def __init__(self):
        super().__init__()
        self.tesseract_configs = ["-c",
                                  "tessedit_char_whitelist=0123456789Ø+*,.=x±RhHSmbKPT()"] + self.tesseract_configs


def ocr(img, psm=11):
    img = erode(np.uint8(img), kernel=2)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2pil(img)
    builder = MyBuilder()
    builder.tesseract_flags = ['--psm', str(psm)]
    word_boxes = tool.image_to_string(
        img,
        lang="pmeng6",
        builder=builder
    )

    img = pil2cv(img)
    for wb in word_boxes:
        (minx, miny), (maxx, maxy) = wb.position
        cv2.rectangle(img, (minx, miny), (maxx, maxy), 128, 3)
        cv2.putText(img, wb.content, (minx, miny - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    1, 128, 2)
    polys_np, labels = zip(*[convert(word_box) for word_box in word_boxes]) if word_boxes else ([], [])

    return polys_np, labels


def rotated_ocr(img, angle, psm=11):
    img = img.rotate(angle, expand=True)
    polys_np, labels = ocr(img, psm)
    return labels

DILATION_SIZE = 2
EROSION_SIZE = 2


def dilate(cv_img, kernel=3):
    return cv2.dilate(cv_img, np.ones((kernel, kernel), np.uint8))


def erode(cv_img, kernel=3):
    return cv2.erode(cv_img, np.ones((kernel, kernel), np.uint8))


def show(path):
    img = cv2.imread(path)
    return Image.fromarray(img)


def recorgnize(img, psm=11):
    polys_np, rec_scores = [], []
    labels = {-90: [], 0: []}
    for i, angle in enumerate([0, -90]):
        labels0 = rotated_ocr(img, angle, psm)  # , builder=builder)
        labels[angle].extend(labels0)
    return labels

def process(img, psm=11):
    img = process_for_getting_only_text(img)
    img = Image.fromarray(img)
    labels = recorgnize(img, psm)
    return labels


def filt(variants):
    return list(filter(lambda v: v != ' ' and any(list(map(str.isdigit, v))), variants))


LINEAR_MIN = 10
LINEAR_MAX = 5000

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
    if len(sizes) > 0:
        maxsize = max(sizes)
    if LINEAR_MIN < maxsize < LINEAR_MAX:
        return maxsize


def extract_sizes(cv_img):
    recognized = {-90: [], 0: []}
    #     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    for pic in crop_conturs(cv_img):
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
