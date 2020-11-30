import numpy as np

def move_to_00(contour):
    """
    Translates upper-left contour point (and all its point)
    to the origin of coordinate plane
    @param contour: dxf contour
    @return: contour: translated dxf contour
    """
    assert isinstance(contour, np.ndarray)
    assert len(contour.shape) == 2
    assert contour.shape[1] == 2

    minx, miny = min(contour[:, 0]), min(contour[:, 1])
    contour -= np.array([minx, miny])
    return contour


def get_contour_size(contour):
    maxx, maxy = max(contour[:, 0]), max(contour[:, 1])
    return maxx, maxy