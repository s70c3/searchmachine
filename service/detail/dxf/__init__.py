import os
import json
import numpy as np
from shapely.geometry import Polygon, Point
from .load import get_contour_from_dxf
from .optimizations import optimize_contour_shapely


class Loader:
    def __init__(self, with_cache, base_path):
        self.with_cache = with_cache
        self.base_dxf_path = base_path
        base_path = base_path[:-1] if base_path.endswith('/') else base_path
        self.base_cache_path = '/'.join(base_path.split('/')[:-1]) + '/jsoned/'

    def _is_cachied(self, name):
        path = self.base_cache_path + name + '.json'
        return os.path.exists(path)

    def _load_from_cache(self, name):
        path = self.base_cache_path + name + '.json'
        points = json.load(open(path))
        points = np.array(points)
        return points

    def _load_from_dxf(self, name):
        path = self.base_dxf_path + name
        points = get_contour_from_dxf(path)
        points = optimize_contour_shapely(points)
        return points

    def _cache_dxf(self, name, points):
        path = self.base_cache_path + name + '.json'
        json.dump(points.tolist(), open(path, 'w'))

    def load_dxf(self, name: str) -> np.array:
        if self.with_cache:
            if self._is_cachied(name):
                points = self._load_from_cache(name)
            else:
                points = self._load_from_dxf(name)
                self._cache_dxf(name, points)
        else:
            points = self._load_from_dxf(name)

        return points

    def get_sizes(self, name):
        dxf = self.load_dxf(name)
        minx, miny = dxf.min(axis=0)
        maxx, maxy = dxf.max(axis=0)
        w, h = maxx - minx, maxy - miny
        return w, h

    def get_square(self, name):
        dxf = self.load_dxf(name)
        points = list(map(Point, dxf.tolist()))
        poly = Polygon(points)
        square = poly.area
        return square

    def get_points(self, name):
        dxf = self.load_dxf(name)
        points = dxf.tolist()
        return points