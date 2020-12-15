from typing import Tuple, List, Optional
import numpy as np
import torch
import pickle
from dataclasses import dataclass
from predict_norms.model import ConvModel
from predict_norms import common
from predict_norms.predict_profilnaya import predict_profilnaya
from predict_norms.predict_gibka import predict_gibka
from predict_norms.predict_termicheskaya import predict_termicheskaya
from predict_operations.predict import pilpaper2operations
from PIL import Image

_profilnaya_operation = 'профильно-вырезная электрофизическая лучевая лазерная'
_gibka_operation = 'гибка'
_termicheskaya_operation = 'термическая резка плазменно-дуговая'

model_pt = './predict_norms/predict_norms.pth'
with open ('./predict_norms/norms_operations.pkl', 'rb') as f:
    operations = pickle.load(f)
model_predict_norms = ConvModel(len(operations))
model_predict_norms.load_state_dict(torch.load(model_pt, map_location='cpu'))
model_predict_norms = model_predict_norms.eval()
with open('./predict_norms/details.pkl', 'rb') as f:
    all_details = pickle.load(f)
with open('./predict_norms/predict_ops.pkl', 'rb') as f:
    model_predict_ops = pickle.load(f)
with open('./predict_norms/predict_ops_massdes.pkl', 'rb') as f:
    model_predict_ops_massdes = pickle.load(f)

@dataclass
class OpsNormsResponse:
    result: List[Tuple]
    error: Optional[str]

def _ops_to_result(ops):
    return [(op, None) for op in ops]

def _ops_norms_to_result(ops, norms):
    return [(op, norms.get(op)) for op in ops]


def _convert_detail_name_to_ohe(curdetail, detail_names):
    r = np.zeros(len(detail_names), dtype=np.uint8)
    r[detail_names.index(curdetail)] = 1
    return r

def _check_tabular(detail_name:str, mass:float, thickness:float, mass_des:float=None):
    if detail_name not in all_details:
        return False, f"Detail {detail_name} is not allowed"
    if mass <= 0:
        return False, f"Mass {mass} <= 0 is not allowed"
    if thickness <= 0:
        return False, f"Thickness {thickness} <= 0 is not allowed"
    if mass_des is not None:
        if mass_des <= 0:
            return False, f"Billet mass {mass_des} <= 0 is not allowed"
        if mass_des < mass:
            return False, f"Billet mass {mass_des} < detail mass {mass} is not allowed"
    return True, ''

def predict_operations_and_norms(img, detail_name:str, mass:float, thickness:float,
                                 length:float = None, width:float = None, mass_des:float=None):
    detail_name = detail_name.lower()
    tabular_is_correct, msg = _check_tabular(detail_name, mass, thickness, mass_des)
    if not tabular_is_correct: return OpsNormsResponse(result=[], error=msg)
    try: projections = common.extract_projections(img)
    except: return OpsNormsResponse(result=[], error="Failed to extract projections from the image")
    try: predicted_ops = _predict_operations(detail_name, mass, thickness, mass_des)
    except Exception as e: return OpsNormsResponse(result=[], error=f"Failed to predict operations with exception {e}")
    try: predicted_norms, msgs = _predict_norms(projections, detail_name, mass, thickness, length, width)
    except Exception as e: return OpsNormsResponse(result=_ops_to_result(predicted_ops), error=f"Failed to predict norms with exception {e}")
    error = None if len(msgs) == 0 else ' '.join(msgs)
    return OpsNormsResponse(result=_ops_norms_to_result(predicted_ops, predicted_norms), error=error)

def _to_pil(img):
    img = np.tile(np.expand_dims(img, axis=-1), reps=np.array((1,1,3)))
    return Image.fromarray(img)

def predict_operations_and_norms_image_only(img):
    try: predicted_ops = pilpaper2operations(_to_pil(img))
    except Exception as e: return OpsNormsResponse(result=[], error=f"Failed to predict operations with exception {e}")
    predicted_ops = [_.lower() for _ in predicted_ops]
    try: projections = common.extract_projections(img)
    except: return OpsNormsResponse(result=_ops_to_result(predicted_ops), error="Failed to extract projections from the image")
    try: predicted_norms = _predict_norms_img_only(projections)
    except Exception as e: return OpsNormsResponse(result=_ops_to_result(predicted_ops), error=f"Failed to predict norms with exception {e}")
    return OpsNormsResponse(result=_ops_norms_to_result(predicted_ops, predicted_norms), error=f"Some norms might be missing because of operations and norm operations mismatch")

def _predict_operations(detail_name:str, mass:float, thickness:float, mass_des:float=None)->np.ndarray:
    """
    Args:
        detail_name (str): name of the detail
        mass (float): mass of the detail
        thickness (float): thickness of the required material
        mass_des (float): mass of the stub for the detail, optional

    Returns: list of operations if successful else `None`
    """
    detail = _convert_detail_name_to_ohe(detail_name, all_details)
    X = np.concatenate((detail, np.array([mass]), np.array([thickness])), axis=0)
    if mass_des is not None:
        X = np.concatenate((X, np.array([mass_des])), axis=0)
    model = model_predict_ops if mass_des is None else model_predict_ops_massdes
    pred = model.predict(X.reshape(1,-1))[0]
    return np.array(operations)[pred == 1].tolist()

def _predict_norms_img_only(img_projections):
    inp = common.projections_to_model_input(img_projections)
    with torch.no_grad():
        output = model_predict_norms(inp)
    output = np.clip(common.np(output), a_min=0, a_max=None)
    return {
        operations[i]: output[0][i] for i in range(len(operations))
    }


def _predict_norms(img_projections, detail_name:str, mass:float, thickness:float, length:float = None, width:float = None):
    '''
    Args:
        img: grayscale image array with values in range 0..255
        length: больший из габаритов
        width: меньший из габаритов
    returns: operations and corresponing norms values if successful else `None`
    '''
    res = _predict_norms_img_only(img_projections)
    msgs = []
    try: res[_profilnaya_operation] = predict_profilnaya(img_projections, mass, detail_name, thickness)
    except: msgs.append(f'Failed to improve `{_profilnaya_operation}`')
    try: res[_gibka_operation] = predict_gibka(img_projections, mass, detail_name, thickness)
    except: msgs.append(f'Failed to improve `{_gibka_operation}`')
    try: res[_termicheskaya_operation] = predict_termicheskaya(img_projections, mass, detail_name, thickness)
    except: msgs.append(f'Failed to improve `{_termicheskaya_operation}`')
    return res, msgs
