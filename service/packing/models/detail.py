import os
import numpy as np
from dataclasses import dataclass
from .dxf_parsing import load_optimized_dxf, load_optimized_json_dxf
from shapely.geometry import Polygon

DXF_BASE_PATH = "/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ/"
JSON_BASE_PATH =  "/data/detail_price/dxf_хпц/jsoned/"

@dataclass
class Detail:
    w: int
    h: int
    quantity: int
    idx: int
    dxf_name: str = None
    dxf_points: list = None

    def get_loading_errors(self):
        # tries to load dxf, returns object with errors
        errors = []
        json_name = self.dxf_name.replace('.DXF', '.json').replace('.dxf', '.json')
        
        if self.dxf_name.lower() == 'none':
            errors.append('dxf not provided. Expected dxf file name, found None')
            return errors
        
        if not os.path.exists(JSON_BASE_PATH + json_name):
            if not os.path.exists(DXF_BASE_PATH + self.dxf_name):
                errors.append('Cant find dxf on disk')
                return errors
            else:
                # load dxf
                try:
                    contour = load_optimized_dxf(DXF_BASE_PATH + self.dxf_name)
                except Exception as e:
                    errors.append('Cant load dxf. Error: ' + str(e))
        else:
            # load from json (faster)
            contour = load_optimized_json_dxf(JSON_BASE_PATH + json_name)

        self.dxf_points = contour
        return errors
        
    def load_dxf_points(self):
        assert self.dxf_name is not None, 'for dxf loading dxf path should be provided'
        # dxf_points are preloaded by get_loading_errors() if it correct. If incorrect, set 
        # default dxf - rect of given size
        if self.dxf_points is not None:
            contour = self.dxf_points # load_optimized_dxf(DXF_BASE_PATH + self.dxf_name)
            print('Contour %s with %d points' % (self.dxf_name, contour.shape[0]))
        else:
            w, h = self.w, self.h
            contour = np.array([(0, 0), (w, 0), (w, h), (0, h)])
            print('Contour %s loading error' % self.dxf_name)
        return contour

    def get_dxf_size(self):
        assert self.dxf_name is not None
        points = self.load_dxf_points()
        maxx, maxy = max(points[:, 0]), max(points[:, 1])
        self.w, self.h = maxx, maxy
        return maxx, maxy
    
    def get_size(self):
        return self.w, self.h

    def rotate(self):
        self.w, self.h = self.h, self.w
    
    def decrease(self, n):
        assert n <= self.quantity
        self.quantity -= n
    
    def fits(self, wh):
        assert isinstance(wh, tuple)
        assert len(wh) == 2
        return max(wh) >= max(self.w, self.h) and min(wh) >= min(self.w, self.h)
    
    def get_detail_square(self):
        if self.dxf_name is not None:
            poly = Polygon(self.load_dxf_points())
            return poly.area
        return self.w * self.h
    
    def get_details_square(self):
        return self.get_detail_square() * self.quantity
    
    def is_rect(self):
        return self.w == self.h
    
    def get_shape(self):
        return self.w, self.h