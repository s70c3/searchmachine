from pathlib import Path
import pickle, torch
import numpy as np
from predict_norms.predict_profilnaya_model import ConvModel
from predict_norms import common

#########################
## hardcoded constants ##
#########################

coef = 0.8
option = ['mass', 'details', 'thickness']
threshold = [0, 0.25, 0.5, 0.75, 1, 2, 100000]
values = [0.105189, 0.348, 0.601408, 0.885801, 1.424103, 3.0432]

#################
## read models ##
#################

_folder = Path('./predict_norms/files/')

def _read_img_model():
    model = ConvModel(1)
    model.load_state_dict(torch.load(_folder / 'gibka.pth', map_location='cpu'))
    model = model.eval()
    return model

def _read_supported_details():
    with open(_folder / 'gibka_supported_details.pkl', 'rb') as f:
        return pickle.load(f)

def _read_clf_cl():
    with open(_folder / 'gibka_clf.pkl', 'rb') as f:
        return pickle.load(f)

_img_model = _read_img_model()
_SUPPORTED_DETAILS = _read_supported_details()
_clf_cl = _read_clf_cl()

#############
## predict ##
#############

def _check_values(mass, detail_name, thickness):
    assert mass > 0, f'mass {mass} must be > 0'
    assert detail_name in _SUPPORTED_DETAILS, f'detail {detail_name} not supported values ({", ".join(_SUPPORTED_DETAILS)})'
    assert thickness > 0, f'thickness {thickness} must be > 0'

def _convert_detail(detail):
    detail_names = _SUPPORTED_DETAILS
    r = np.zeros(len(detail_names), dtype=np.uint8)
    r[detail_names.index(detail)] = 1
    return r

def _predict_tabular(mass, detail_name, thickness):
    detail = _convert_detail(detail_name)
    inp = np.concatenate((np.array([mass]),detail,np.array([thickness])), axis=0)
    inp = np.expand_dims(inp, axis=0)
    pred_class = _clf_cl.predict(inp)
    return values[int(pred_class[0])]

def _predict_img(img):
    inp = common.img_to_model_input(img)
    with torch.no_grad():
        output = _img_model(inp)
    return np.clip(common.np(output), a_min=0, a_max=None)[0][0]

def predict_gibka(img, mass, detail_name, thickness):
    _check_values(mass, detail_name, thickness)
    tab_pred = _predict_tabular(mass, detail_name, thickness)
    img_pred =_predict_img(img)
    return coef * tab_pred + (1 - coef) * img_pred

