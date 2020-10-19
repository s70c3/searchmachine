from PIL import Image
import numpy as np

import cv2
import pyocr
import pyocr.builders

from .detection import crop_conturs, pil2cv
from imutils.object_detection import non_max_suppression

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


from PIL import Image
import numpy as np

import cv2
import pyocr
import pyocr.builders

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


class MyBuilder(pyocr.builders.WordBoxBuilder):
    def __init__(self):
        super().__init__()
        self.tesseract_configs = ["-c", "tessedit_char_whitelist=0123456789+*,.=x±RhHS()"] + self.tesseract_configs


def ocr(img, psm=11):
    builder = MyBuilder()
    builder.tesseract_flags = ['--psm', str(psm)]

    #should be replaced with tunedeng.traineddata to be better
    word_boxes = tool.image_to_string(
        img,
        lang="eng",
        builder=builder
    )

    polys_np, labels = zip(*[convert(word_box) for word_box in word_boxes]) if word_boxes else ([], [])
    #     print(labels)
    #     show_one(img1)
    return polys_np, labels


def rotated_ocr(img, angle, builder=pyocr.builders.WordBoxBuilder(), psm=11):
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
    #     img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    #     img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = erode(dilate(img, DILATION_SIZE), EROSION_SIZE)
    #     img = cv2.medianBlur(img, 3)
    img = Image.fromarray(img)
    labels = recorgnize(img, psm)
    return labels


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
    for i in range(len(sizes)):
        if sizes[i] > 5000 and sizes[i] % 100 // 10 == 4 or sizes[i] % 100 // 10 == 2:
            sizes[i] = sizes[i] // 100
    maxsize = max(sizes)
    if LINEAR_MIN < maxsize < LINEAR_MAX:
        return maxsize


def extract_sizes(pil_img):
    recognized = {-90: [], 0: []}
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    for pic in crop_conturs(cv_img):
        projection_recognized = process(pic)
        for angled in projection_recognized:
            recognized[angled] += filt(projection_recognized[angled])

        projection_recognized_east = process_east(pic)
        for angled in projection_recognized_east:
            temp = filt(projection_recognized_east[angled])
            for s in temp:
                if not s in recognized[angled]:
                    recognized[angled].append(s)

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


def hough_cleaning(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(img,(x1-10,y1),(x2+10,y2),(255,255,255),3)
    return img


def get_img_with_padding(image):
    COLOR = (255, 255, 255)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    (H, W) = image.shape[:2]
    pad_w = ((W+31)//32*32-W)//2
    pad_h = ((H+31)//32*32-H)//2
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w,pad_w, cv2.BORDER_CONSTANT, None, COLOR)
    return image


def get_sizes_boxes(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = erode(dilate(image, 2), 2)
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # load the pre-trained EAST model for text detection
    net = cv2.dnn.readNet("./frozen_east_text_detection.pb")

    # The following two layer need to pulled from EAST model for achieving this.
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass the blob from the image to get the desired output layers
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val, overlapThresh=0.1)
    new_boxes = []
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX - 20) % W
        startY = int(startY - 20) % H
        endX = int(endX + 20) % W
        endY = int(endY + 20) % H
        new_boxes.append((startX, startY, endX, endY))
    return new_boxes


def cv2pil(cv_img):
    return Image.fromarray(cv_img)


def get_text_from_boxes(img, boxes):
    COLOR = (255, 255, 255)
    labels = {-90: [], 0: []}
    for (startX, startY, endX, endY) in boxes:
        # extract the region of interest
        r = img[startY // 2:endY // 2, startX // 2:endX // 2]
        # make margins for tesseract
        r = cv2.copyMakeBorder(r, 200, 200, 200, 200, cv2.BORDER_CONSTANT, None, COLOR)
        r = erode(r, kernel=2)
        r = hough_cleaning(r)
        r = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
        #         show_one(r)

        text = recorgnize(cv2pil(r), psm=8)
        for i, angle in enumerate([0, -90]):
            labels[angle].extend(text[angle])
    return labels


def process_east(img):
    img = get_img_with_padding(img)
    boxes = get_sizes_boxes(img)
    sizes = get_text_from_boxes(img, boxes)
    return sizes


def extract_sizes_with_east(img):
    recognized = {-90: [], 0: []}
    #     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    for pic in crop_conturs(img):
        projection_recognized = process_east(pic)
        for angled in projection_recognized:
            recognized[angled] += filt(projection_recognized[angled])

    recog_linears = {}
    for angled in recognized:
        recog_linears[angled] = list(map(get_linear_size, recognized[angled]))
        recog_linears[angled] = list(filter(lambda v: v is not None, recog_linears[angled]))

    # return a tuple with 2 sizes (height and width) and all sizes
    return recog_linears


# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < 0.1:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return boxes, confidence_val