import cv2
import torch
import predict_operations.detection


def _preprocess(img):
    img = cv2.resize(img, (150,150))
    img = img / 255.
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def np(t): return t.cpu().detach().numpy()

def extract_projections(img):
    return predict_operations.detection.crop_conturs(img)

def projections_to_model_input(projections):
    projections = list(map(_preprocess, projections))
    return [projections]
