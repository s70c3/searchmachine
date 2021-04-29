from .extract_cells import get_cell_imgs
from .predict import predict_mass, predict_material, predict_name, predict_detail


def _make_response(mass, material, name, detail):
    return {
        'mass': mass,
        'material': material,
        'name': name,
        'detail': detail
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
        mass, _, name, detail, material = get_cell_imgs(img)
        mass = _apply_safe(predict_mass, mass)
        material = _apply_safe(predict_material, material)
        name = _apply_safe(predict_name, name)
        detail = _apply_safe(predict_detail, detail)
        return _make_response(mass, material, name, detail)
    except:
        return _make_response(None, None, None, None)