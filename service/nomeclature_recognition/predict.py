import nomeclature_recognition.parsing as parsing
from nomeclature_recognition.predict_utils import *


def _get_img_preprocess_tfms():
    return [create_cut_border_tfm(), create_threshold_tfm(None)]


def predict_mass(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    return parsing.parse_word(img, 'eng', psm=4, oem=None)


def predict_material(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    return parsing.parse_word(img, 'rus', psm=4, oem=None)
