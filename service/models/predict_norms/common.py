import cv2
import torch
from service.models.predict_operations import detection


def _preprocess(img):
    img = cv2.resize(img, (150,150))
    img = img / 255.
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def np(t): return t.cpu().detach().numpy()

def extract_projections(img):
    return detection.crop_conturs(img)

def projections_to_model_input(projections):
    projections = list(map(_preprocess, projections))
    return [projections]
