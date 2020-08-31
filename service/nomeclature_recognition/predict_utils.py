import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import nomeclature_recognition.utils as u


@dataclass
class NamedTfm:
    name: str
    short_name: str
    fn: [[np.ndarray], np.ndarray]

def apply_tfms(img, tfms:List[NamedTfm]):
    for tfm in tfms:
        img = tfm.fn(img)
    return img

def _cut_border(img): return img[5:-5, 5:-5]

def create_cut_border_tfm():
    return NamedTfm(name='cut border 5', short_name='cut', fn=_cut_border)


def create_threshold_tfm(th):
    if th is None:
        return NamedTfm(name=f'threshold=No', short_name=f'threshNo', fn=lambda img: u.identity(img))
    return NamedTfm(name=f'threshold={th}', short_name=f'thresh{th}', fn=lambda img: u.threshold(img, th))
