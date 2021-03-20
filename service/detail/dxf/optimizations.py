import numpy as np
from shapely.geometry import Polygon

def optimize_contour_shapely(points):
    fig = Polygon(list(points))
    fig = fig.simplify(tolerance=5, )
    xs, ys = fig.exterior.coords.xy
    coords = list(zip(list(xs), list(ys)))
    coords = np.array(coords)
    return coords