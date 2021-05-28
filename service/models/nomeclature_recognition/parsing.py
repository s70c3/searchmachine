import pyocr
from PIL import Image
import cv2
import pyocr.builders
from .predict_utils import *



def _check_lang(lang, tool):
    langs = lang.split('+')
    for l in langs:
        if l not in tool.get_available_languages():
            raise Exception(f'Language {l} not installed in tesseract. Available languages: {tool.get_available_languages()}')


def parse_words_with_location(img, lang, psm, oem=None, only_digits=False):
    tool = pyocr.get_available_tools()[0]
    _check_lang(lang, tool)
    img = u.gray2rgb(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    builder = pyocr.builders.WordBoxBuilder()
    flags = ['--psm', str(psm)]
    if oem is not None:
        flags += ['--oem', str(oem)]
    if only_digits:
        flags += ['nobatch', 'digits']
    builder.tesseract_flags = flags
    word_boxes = tool.image_to_string(img, lang=lang, builder=builder)
    return word_boxes


def parse_word(img, lang, psm=8, oem=None, only_digits=False):
    word_boxes = parse_words_with_location(img, lang, psm, oem, only_digits)
    text = [wb.content for wb in word_boxes]
    return "|".join(text)
