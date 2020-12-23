import numpy as np
import json
from shapely.geometry import Polygon
from .parsing import get_contour_from_dxf
from . import compressing
from .utils import move_to_00

def optimize_contour(points):
    # list of tuples of 2 coords
    points = np.array(list(map(list, points)))
    points = compressing.points_sequence_to_lines(points)
    points = compressing.filter_points_duplicates(points)
#     points = compressing.skew_lines_to_pair_points(points)
    points = move_to_00(points)
    return points

def optimize_contour_shapely(points):
    fig = Polygon(list(points))
    fig = fig.simplify(tolerance=5, )
    xs, ys = fig.exterior.coords.xy
    coords = list(zip(list(xs), list(ys)))
    coords = np.array(coords)
    return coords

def load_optimized_dxf(path):
    dxf = get_contour_from_dxf(path)
    dxf = optimize_contour_shapely(dxf)
    return dxf


def load_optimized_json_dxf(path):
    dxf = json.load(open(path))
    dxf = np.array(dxf)
    dxf = optimize_contour(dxf)
    return dxf