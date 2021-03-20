from PIL import Image
from typeguard import check_argument_types
from service.models.base import BaseModel
from .ocr import extract_sizes


class LinearSizesModel(BaseModel):
    @staticmethod
    def predict(img: Image):
        assert check_argument_types()

        sizes = extract_sizes(img)[1]
        return sizes
