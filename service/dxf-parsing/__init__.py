from .parsing import get_contour_from_dxf
from .compressing import points_sequence_to_lines, filter_points_duplicates
from .utils import move_to_00

def load_optimized_dxf(path):
    dxf = get_contour_from_dxf(path)
    dxf = filter_points_duplicates(points_sequence_to_lines(dxf))
    dxf = move_to_00(dxf)
    return dxf