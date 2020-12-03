import pdf2image
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from predict_norms import api
import yaml
from collections import OrderedDict

###########
## utils ##
###########

def pil2cv(pil_img):
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)

def cv2pil(cv_img):
    return Image.fromarray(cv_img)

def read_pdf(pt):
    img = pdf2image.convert_from_path(pt)[0]
    return pil2cv(img)

def read_gray(pt):
    pt = Path(pt)
    if pt.suffix == '.pdf':
        img = read_pdf(pt)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(str(pt), cv2.IMREAD_GRAYSCALE)
        if not img.dtype == np.uint8:
            print('Alert')
    return img

# ##########
# ## main ##
# ##########

_default_length = 100
_profilnaya_operation = 'профильно-вырезная электрофизическая лучевая лазерная'


def _check_pred(name, filepath, detail, mass ,thickness, ops, profilnaya_norm):
    prefix = f'Test {name}: '
    img = read_gray(filepath)
    ops = ops.split('|')
    ops_pred = api.predict_operations(detail, mass, thickness)
    norms_pred = api.predict_norms(img, detail, mass, thickness, _default_length, _default_length)
    profilnaya_norm_pred = norms_pred[_profilnaya_operation]
    assert set(ops_pred) == set(ops), prefix + "Incorrect ops predicted"
    assert np.abs(
        profilnaya_norm_pred - profilnaya_norm) < 0.05, prefix + f"Incorrect norm predicted for {_profilnaya_operation}"
    assert np.all(np.array(list(norms_pred.values())) >= 0), prefix + "Norm predictions must be >= 0"
    comb_pred = api.predict_operations_and_norms(img, detail, mass, thickness,_default_length, _default_length)
    assert (isinstance(comb_pred, (list,tuple))
            and isinstance(comb_pred[0], (list,tuple))
            and all([len(el) == 2 for el in comb_pred])), "`predict_operations_and_norms` must return list of pairs (operation, norm)"
    comb_pred = OrderedDict(comb_pred)
    assert list(comb_pred.keys()) == ops_pred, 'combined prediction operations do not match operations prediction'
    assert all([v == norms_pred.get(k) for (k,v) in comb_pred.items()]), 'combined prediction norms do not match norms prediction'


def main():
    with open('unittests/predict_norms_tests.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for test in data:
        _check_pred(**test)

