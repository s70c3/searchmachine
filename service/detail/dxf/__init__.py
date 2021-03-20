import os
import json
from typeguard import check_argument_types
import numpy as np
from .load import get_contour_from_dxf
from .optimizations import optimize_contour_shapely

DXF_PATH = "/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ/"
CACHE_DXF_PATH =  "/data/detail_price/dxf_хпц/jsoned/"


class Loader:
    def __init__(self, with_cache):
        self.with_cache = with_cache

    def _is_cachied(self, name):
        path = CACHE_DXF_PATH + name + '.json'
        return os.path.exists(path)

    def _load_from_cache(self, name):
        path = CACHE_DXF_PATH + name + '.json'
        points = json.load(open(path))
        points = np.array(points)
        return points

    def _load_from_dxf(self, name):
        path = DXF_PATH + name
        points = get_contour_from_dxf(path)
        points = optimize_contour_shapely(points)
        return points

    def _cache_dxf(self, name, points):
        path = CACHE_DXF_PATH + name + '.json'
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

