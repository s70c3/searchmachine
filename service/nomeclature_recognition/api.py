from nomeclature_recognition.extract_cells import get_cell_imgs
from nomeclature_recognition.predict import predict_mass, predict_material


def _make_response(mass, material):
    return {
        'mass': mass,
        'material': material
    }

def _apply_safe(fn, arg):
    try:
        return fn(arg)
    except:
        return None


def extract_nomenclature(img):
    '''
    @param img: grayscale image of the drawing
    '''
    try:
        mass, _, material = get_cell_imgs(img)
        mass = _apply_safe(predict_mass, mass)
        material = _apply_safe(predict_material, material)
        return _make_response(mass, material)
    except:
        return _make_response(None, None)
