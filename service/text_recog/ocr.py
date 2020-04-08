from PIL import Image

import cv2
import pyocr
import pyocr.builders

import numpy as np

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


def ocr(img, psm=11):
    builder = pyocr.builders.WordBoxBuilder()
    # builder.tesseract_flags = ['--psm','1']
    builder.tesseract_flags = ['--psm',str(psm)]
    word_boxes = tool.image_to_string(
        img,    
        lang="eng",
        builder=builder
    )
    
    polys_np, labels = zip(*[ convert(word_box) for word_box in word_boxes]) if word_boxes else ([], [])
    
    return polys_np, labels


def rotated_ocr(img, angle, builder=pyocr.builders.WordBoxBuilder()):
    img = img.rotate(angle, expand=True)
    polys_np, labels = ocr(img)
    return labels



DILATION_SIZE = 2
EROSION_SIZE = 4
    
def dilate(cv_img, kernel=2):
    return cv2.dilate(cv_img, np.ones((kernel, kernel), np.uint8))

def erode(cv_img, kernel=2):
    return cv2.erode(cv_img, np.ones((kernel, kernel), np.uint8))


def show(path):
    img =  cv2.imread(path)
    return Image.fromarray(img)


def recorgnize(img):
    polys_np, labels, rec_scores = [],[],[]

    for i, angle in enumerate([0, -90]):
        labels0 = rotated_ocr(img, angle)#, builder=builder)
        labels.extend(labels0)
    return labels


def process(pil_img):
    img =  cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    img = erode(dilate(img, DILATION_SIZE), EROSION_SIZE)
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
        maxsize = max(sizes)
        if maxsize > LINEAR_MIN and maxsize < LINEAR_MAX:
            return maxsize
        
        
    recognized = filt(process(pil_img))

    recog_linears = list(map(get_linear_size, recognized))
    recog_linears = list(filter(lambda v: v is not None, recog_linears))
    
    return recog_linears