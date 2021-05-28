from . import parsing
from .predict_utils import *
from .predict_material import *
import difflib
import pickle
from yamlparams.utils import Hparam
import os

def _read_list_of_detail_names():
    with open(f'{os.path.dirname(__file__)}/files/det_joined.pkl', 'rb') as f:
        return pickle.load(f)


def _get_img_preprocess_tfms():
    return [create_cut_border_tfm(), create_threshold_tfm(None)]


def predict_mass(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    return parsing.parse_word(img, 'eng', psm=4, oem=None, only_digits=True)

_allowed_name_symbols = (list(map(str, range(0,10))) + list(map(chr, range(ord('a'), ord('z') + 1)))
               + list(map(chr, range(ord('а'), ord('я') + 1))) + ['.', ','])
def _filter_name_symbols(val):
    val = [v for v in val if v in _allowed_name_symbols]
    return "".join(val)


def predict_name(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    word = parsing.parse_word(img, 'rus', psm=4, oem=None)
    return _filter_name_symbols(word)


def predict_detail(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    candidate = parsing.parse_word(img, 'rus', psm=4, oem=None).lower()
    return difflib.get_close_matches(candidate, _details, n=1)[0]



def predict_material(cell_img):
    img = apply_tfms(cell_img, _get_img_preprocess_tfms())
    wboxes = parsing.parse_words_with_location(img, 'rus', psm=11, oem=None)
    if len(wboxes) > 0:
        ut, lt = combine_text(wboxes)
        material = f"{ut} / {lt}"
    else:
        material = None
    return material

# config = Hparam('./config.yml')

# if config.run.models.nomenclature:
#     #################
#     ## read models ##
#     #################
_details = _read_list_of_detail_names()
