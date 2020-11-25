import cv2
import torch
import predict_operations.detection

def _extract_projections(img):
    return predict_operations.detection.crop_conturs(img)

def _preprocess(img):
    img = cv2.resize(img, (150,150))
    img = img / 255.
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def np(t): return t.cpu().detach().numpy()

def img_to_model_input(img):
    projections = _extract_projections(img)
    projections = list(map(_preprocess, _extract_projections(img)))
    return [projections]