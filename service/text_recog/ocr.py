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
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    (H, W) = image.shape[:2]
    pad_w = ((W + 31) // 32 * 32 - W) // 2
    pad_h = ((H + 31) // 32 * 32 - H) // 2
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, None, COLOR)

    return image


def get_sizes_boxes(image):
    (H, W) = image.shape[:2]
    print("sizes", H, W)
    if H < 1800 and W < 1800:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        f = 2
    elif H > 3500 and W > 3500:
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        (H, W) = image.shape[:2]
        COLOR = (255, 255, 255)
        image = cv2.copyMakeBorder(image, 0, H % 32, 0, W % 32, cv2.BORDER_CONSTANT, None, COLOR)
        f = 0.5
    elif H > 5000 or W > 5000:
        return [], 1
    else:
        f = 1

    # img_copy_for_boxes = image.copy()
    kernel = np.uint((5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = hough_cleaning(image)
    image = erode(dilate(erode(image, 7), 4), 7)

    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # load the pre-trained EAST model for text detection
    net = cv2.dnn.readNet("../size_detecting_utils/frozen_east_text_detection.pb")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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
        startX = int(startX - 50) % W
        startY = int(startY - 50) % H
        endX = int(endX + 50) % W
        endY = int(endY + 50) % H
        new_boxes.append((startX, startY, endX, endY))
    # for (startX, startY, endX, endY) in new_boxes:
    #     cv2.rectangle(img_copy_for_boxes, (startX, startY), (endX, endY), (0, 0, 255), 1)

    # show_one(img_copy_for_boxes)
    return new_boxes, f

def get_image_with_boxes(img, boxes, f):
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    COLOR = (255, 255, 255)
    img = pil2cv(img)
    img = hough_cleaning(img)
    kernel = np.uint((5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    mask = np.ones(img.shape)*255
    for (startX, startY, endX, endY) in boxes:
        #extract the region of interest
        mask[int(startY//f):int(endY//f), int(startX//f):int(endX//f)] = img[int(startY//f):int(endY//f), int(startX//f):int(endX//f)]
    img = cv2pil(img)
    return mask

def process_east_for_cleaning(img):
    img = get_img_with_padding(img)
    boxes, flag  = get_sizes_boxes(img)
    mask = get_image_with_boxes(img, boxes, flag)
    return mask

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
            if scoresData[i] < 0.00001:
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


def yolo_prepare():
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)

    # derive the paths to the YOLO weights and model configuration
    weightsPath = '../size_detecting_utils/custom-yolov4-detector_best.weights'
    configPath = '../size_detecting_utils/custom-yolov4-detector.cfg'
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln


def clean_plusminus(image, net, ln):
    import time
    labelsPath = '../size_detecting_utils/obj.names'
    COLORS = np.random.randint(0, 255, size=(1, 3), dtype="uint8")
    # load the COCO class labels our YOLO model was trained on
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    img_gray = pil2cv(image)
    tolerance_mask = np.ones(img_gray.shape) * 255
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x + 3, y - 10), (x + w + 50, y + h), (255, 255, 255), -1)
            #get the mask with tolerances
            tolerance_mask[y - 10:y + h + 10, x - 100:x + w + 100] = img_gray[y - 10:y + h + 10, x - 100:x + w + 100]

    else:
        tolerance_mask = None
    return image, tolerance_mask


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
    net, ln = yolo_prepare()
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    image, tolerance_mask = clean_plusminus(img, net, ln)
    img = cv2pil(img)

    builder = MyBuilder()
    builder.tesseract_flags = ['--psm', str(psm)]
    word_boxes = tool.image_to_string(
        img,
        lang="pmeng6",
        builder=builder
    )
    if tolerance_mask is not None:
        tolerance_boxes = tool.image_to_string(
            cv2pil(tolerance_mask),
            lang="pmeng6",
            builder=builder
        )
        for wb in tolerance_boxes:
            (minx, miny), (maxx, maxy) = wb.position
            cv2.rectangle(tolerance_mask, (minx, miny), (maxx, maxy), 128, 3)
            cv2.putText(tolerance_mask, wb.content, (minx, miny - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 128, 2)
            # show_one(tolerance_mask)
    else:
        tolerance_boxes = []

    img = pil2cv(img)
    img = np.uint8(img)
    for wb in word_boxes:
        (minx, miny), (maxx, maxy) = wb.position
        cv2.rectangle(img, (minx, miny), (maxx, maxy), 128, 3)
        cv2.putText(img, wb.content, (minx, miny - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    1, 128, 2)

    polys_np, labels = zip(*[convert(word_box) for word_box in word_boxes]) if word_boxes else ([], [])

    polys_np_t, labels_tolerance = zip(*[convert(word_box) for word_box in tolerance_boxes]) if tolerance_boxes else (
    [], [])

    return polys_np, labels, labels_tolerance


def rotated_ocr(img, angle, builder=pyocr.builders.WordBoxBuilder(), psm=11):
    img = img.rotate(angle, expand=True)
    polys_np, labels, labels_tolerance = ocr(img, psm)
    return labels, labels_tolerance


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
    labels_t = []
    for i, angle in enumerate([0, -90]):
        labels0, labels_t0 = rotated_ocr(img, angle, psm)  # , builder=builder)
        labels[angle].extend(labels0)
        labels_t.extend(labels_t0)
    return labels, labels_t


def process(img, psm=11):
    img = process_east_for_cleaning(img)
    # show_one(img)
    img = Image.fromarray(img)
    labels, labels_t = recorgnize(img, psm)
    return labels, labels_t


def filt(variants):
    return list(filter(lambda v: v != ' ' and any(list(map(str.isdigit, v))), variants))


LINEAR_MIN = 10
LINEAR_MAX = 5000


def get_linear_size(w):
    if len(w) == 0:
        return None

    if not any(list(map(str.isdigit, w))):
        return None
    # if 'R' in w or 'K' in w or 'T' in w or 'P' in w:
    #     return None
    if 'R' in w:
        return None
    if w.count('.') > 2:
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


def get_tolerances(sizes):
    tolerances_t = []
    for size in sizes:
        if '±' in size:
            t = size.split('±')[1]
            tolerances_t.append(t)
    return tolerances_t


def extract_sizes(cv_img):
    recognized = {-90: [], 0: []}
    tolerances = []
    for pic in crop_conturs(cv_img):
        projection_recognized, tolerances_labels = process(pic)
        for angled in projection_recognized:
            recognized[angled] += filt(projection_recognized[angled])
        tolerances.extend(get_tolerances(tolerances_labels))
    recog_linears = {}
    for angled in recognized:
        tolerance = get_tolerances(recognized[angled])
        tolerances.extend(tolerance)
        # print(tolerances)
        recog_linears[angled] = list(map(get_linear_size, recognized[angled]))
        recog_linears[angled] = list(filter(lambda v: v is not None, recog_linears[angled]))

    linears = []
    all_sizes = []
    for angled in recog_linears:
        all_sizes.extend(recog_linears[angled])
        if len(recog_linears[angled]) > 0:
            linears.append(max(recog_linears[angled]))
    # return a tuple with 2 sizes (height and width) and all sizes
    return linears, all_sizes, tolerances
