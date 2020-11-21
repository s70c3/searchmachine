import numpy as np
import cv2
import torch
import pickle
import predict_operations.detection
from predict_norms.model import ConvModel


model_pt = './predict_norms/predict_norms.pth'
with open ('./predict_norms/norms_operations.pkl', 'rb') as f:
    operations = pickle.load(f)
model_predict_norms = ConvModel(len(operations))
model_predict_norms.load_state_dict(torch.load(model_pt))
model_predict_norms = model_predict_norms.eval()
with open('./predict_norms/details.pkl', 'rb') as f:
    all_details = pickle.load(f)
with open('./predict_norms/predict_ops.pkl', 'rb') as f:
    model_predict_ops = pickle.load(f)
with open('./predict_norms/predict_ops_massdes.pkl', 'rb') as f:
    model_predict_ops_massdes = pickle.load(f)



def _extract_projections(img):
    return predict_operations.detection.crop_conturs(img)

def _preprocess(img):
    img = cv2.resize(img / 255., (150,150))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def _np(t): return t.cpu().detach().numpy()

def _convert_detail_name_to_ohe(curdetail, detail_names):
    r = np.zeros(len(detail_names), dtype=np.uint8)
    r[detail_names.index(curdetail)] = 1
    return r

def predict_operations(detail_name:str, mass:float, thickness:float, mass_des:float=None)->np.ndarray:
    """
    Args:
        detail_name (str): name of the detail
        mass (float): mass of the detail
        thickness (float): thickness of the required material
        mass_des (float): mass of the stub for the detail, optional

    Returns: list of operations if successful else `None`
    """
    try:
        print(all_details)
        detail = _convert_detail_name_to_ohe(detail_name, all_details)
        X = np.concatenate((detail, np.array([mass]), np.array([thickness])), axis=0)
        if mass_des is not None:
            X = np.concatenate((X, np.array([mass_des])), axis=0)
        model = model_predict_ops if mass_des is None else model_predict_ops_massdes
        pred = model.predict(X.reshape(1,-1))[0]
        return np.array(operations)[pred == 1].tolist()
    except:
        return None

def predict_norms(img):
    '''
    Args:
        img: grayscale image array with values in range 0..255
    returns: operations and corresponing probabilities if successful else `None`
    '''
    try:
        projections = list(map(_preprocess, _extract_projections(img)))
        with torch.no_grad():
            output = model_predict_norms([projections])
        output = np.clip(_np(output), a_min=0, a_max=None)
        return {
            operations[i]: output[0][i] for i in range(len(operations))
        }
    except:
        return None
