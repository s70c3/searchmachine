import numpy as np
from typeguard import check_argument_types
from service.models.base import BaseModel
from .api import extract_nomenclature


class NomenclatureModel(BaseModel):
    @staticmethod
    def _validate_mass(mass: str) -> float or None:
        if mass is None:
            return mass
        if ',' in mass:
            mass = mass.replace(',', '.')
        try:
            mass = float(mass)
        except ValueError:
            return None
        return mass

    @staticmethod
    def _validate_material(material: str) -> str or None:
        if material is None:
            return material

        materials = ['жесть', 'круг', 'лента', 'лист', 'петля', 'проволока', 'прокат', 'профиль', 'рулон', 'сетка']
        for allowed_material in materials:
            if allowed_material in material:
                return allowed_material
        return material

    @staticmethod
    def predict(img: np.array) -> dict:
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

        prediction = extract_nomenclature(img)
        prediction['mass'] = NomenclatureModel._validate_mass(prediction['mass'])
        prediction['material'] = NomenclatureModel._validate_material(prediction['material'])
        return prediction
