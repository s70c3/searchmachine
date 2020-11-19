import numpy as np
import cv2
import torch
import pickle
import predict_operations.detection
from predict_norms.model import ConvModel


model_pt = './predict_norms/predict_norms.pth'
with open ('./predict_norms/norms_operations.pkl', 'rb') as f:
    operations = pickle.load(f)
model = ConvModel(len(operations))
model.load_state_dict(torch.load(model_pt))
model = model.eval()


def _extract_projections(img):
    return predict_operations.detection.crop_conturs(img)

def _preprocess(img):
    img = cv2.resize(img / 255., (150,150))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def _np(t): return t.cpu().detach().numpy()

def predict_norms(img):
    '''
    Args:
        img: grayscale image array with values in range 0..255
    returns: operations and corresponing probabilities
    '''
    projections = list(map(_preprocess, _extract_projections(img)))
    with torch.no_grad():
        output = model([projections])
    output = np.clip(_np(output), a_min=0, a_max=None)
    return {
        operations[i]: output[0][i] for i in range(len(operations))
    }
