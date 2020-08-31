import pyocr
from PIL import Image
import cv2
import pyocr.builders
from nomeclature_recognition.predict_utils import *


def parse_word(img, lang, psm=8, oem=None):
    tool = pyocr.get_available_tools()[0]
    if lang not in tool.get_available_languages():
        raise Exception(f'Language {lang} not installed in tesseract')
    img = u.gray2rgb(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    builder = pyocr.builders.WordBoxBuilder()
    flags = ['--psm', str(psm)]
    if oem is not None:
        flags += ['--oem', str(oem)]    
    builder.tesseract_flags = flags
    word_boxes = tool.image_to_string(img, lang=lang, builder=builder)
    text = [wb.content for wb in word_boxes]
    return "|".join(text)
