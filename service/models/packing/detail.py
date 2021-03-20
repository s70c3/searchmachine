import os
import json
import numpy as np
from dataclasses import dataclass
from service.detail.dxf import Loader
from shapely.geometry import Polygon, Point


class Dxf:
    def __init__(self, name):
        self.name = name
        self.rotated = False

    def load(self):
        try:
            self.loader = Loader(with_cache=True)
            self.points = self.loader.load_dxf(self.name)
            self.w, self.h = self._get_sizes(self.points)

        except Exception as e:
            print(f'Countour {self.name} loading error')
            raise e

    def rotate_(self):
        self.rotated = not self.rotated
        self.load()
        if self.rotated:
            self.w, self.h = self.h, self.w
            # swap xs and ys
            self.points = np.array(list(map(np.transpose, self.points)))

    def get_points(self):
        return self.points

    def _get_sizes(self, points: np.array):
        # assert check_argument_types()
        w = max(points[:, 0]) - min(points[:, 0])
        h = max(points[:, 1]) - min(points[:, 1])
        return w, h


class DetailBase:
    def __init__(self):
        raise NotImplementedError

    def get_square(self):
        raise NotImplementedError

    def get_size(self):
        return self.w, self.h

    def get_shape(self):
        # legacy
        return self.get_size()

    def rotate_(self):
        self.w, self.h = self.h, self.w

    def decrease_(self, n):
        assert n <= self.quantity
        self.quantity -= n

    def fits(self, wh):
        assert isinstance(wh, tuple)
        assert len(wh) == 2
        return max(wh) >= max(self.w, self.h) and min(wh) >= min(self.w, self.h)

    def get_square_x_quantity(self):
        return self.get_square() * self.quantity


class DetailRect(DetailBase):
    def __init__(self, w, h, q, idx, dxf_name=None):
        self.w = w
        self.h = h
        self.quantity = q
        self.idx = idx
        self.dxf = Dxf(dxf_name)

    def get_square(self):
        return self.w * self.h

    def is_rect(self):
        return self.w == self.h


class DetailPolygonal(DetailBase):
    def __init__(self, q, idx, dxf_name=None):
        self.quantity = q
        self.idx = idx
        self.square = None
        self.kim = None

        self.dxf = Dxf(dxf_name)
        self.dxf.load()
        self.w, self.h = self._calc_wh(self.dxf)
        print(f'Contour {self.dxf.name} ok  kim {round(self.get_kim(), 2)} points {len(self.dxf.get_points())}')


    def rotate_(self):
        self.w, self.h = self.h, self.w
        self.dxf.rotate_()

    def to_nest4j_format(self):
        points = self.dxf.get_points().tolist()
        shape = [{'x': x, 'y': y} for (x, y) in points]
        return {'points': shape, 'id': None}

    def get_square(self):
        if self.square is None:
            points = self.dxf.get_points()
            points = list(map(Point, points.tolist()))
            poly = Polygon(points)
            self.square = poly.area
        return self.square

    def get_kim(self):
        if self.kim is None:
            w, h = self.get_size()
            self.kim = self.get_square() / (w*h)
        return self.kim

    def get_square_x_quantity(self):
        return self.get_square() * self.quantity

    def _calc_wh(self, dxf):
        points = dxf.get_points()
        w = max(points[:, 0]) - min(points[:, 0])
        h = max(points[:, 1]) - min(points[:, 1])
        return w, h
