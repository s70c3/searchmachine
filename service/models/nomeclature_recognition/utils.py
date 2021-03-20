import pdf2image
import cv2
import numpy as np
from PIL import Image


def pil2cv(pil_img):
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)

def cv2pil(cv_img):
    return Image.fromarray(cv_img)

def stats(arr, unique=False):
    print(f'shape={arr.shape}, type={arr.dtype}, min={arr.min()}, max={arr.max()}')
    if unique:
        print('unique values: ', np.unique(arr))

def read_pdf(pt):
    img = pdf2image.convert_from_path(pt)[0]
    return pil2cv(img)

def threshold(img, th):
    return (img > th).astype(np.uint8) * 255

def identity(x):
    return x

def invert(img):
    return 255 - img

def gray2rgb(i): return cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)

def extract_contours(img):
    H, W = img.shape

    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,254,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)

    #inverting the image 
    img_bin = invert(img_bin)

    # Length(width) of kernel as 100th of minimal dimension
    kernel_len = min(H, W)//100
    # # Length(width) of kernel as 100th of height
    # kernel_len = H//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    t_vh = threshold(img_vh, 10)
    return t_vh

