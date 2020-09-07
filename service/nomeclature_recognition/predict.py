import nomeclature_recognition.parsing as parsing
from nomeclature_recognition.predict_utils import *
from nomeclature_recognition.predict_material import *


def _get_img_preprocess_tfms():
    return [create_cut_border_tfm(), create_threshold_tfm(None)]


def predict_mass(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    return parsing.parse_word(img, 'eng', psm=4, oem=None)


def predict_material(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    wboxes = parsing.parse_words_with_location(img, 'rus', psm=11, oem=None)
    if len(wboxes) > 0:
        ut, lt = combine_text(wboxes)
        material = f"{ut} / {lt}"
    else:
        material = None
    return material

