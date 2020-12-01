from dataclasses import dataclass
from .dxf_parsing import load_optimized_dxf
from shapely.geometry import Polygon

DXF_BASE_PATH = "/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ/"

@dataclass
class Detail:
    w: int
    h: int
    quantity: int
    idx: int
    dxf_name: str = None

    def load_dxf_points(self):
        assert self.dxf_name is not None, 'for dxf loading dxf path should be provided'
        # default dxf normalization
        contour = load_optimized_dxf(DXF_BASE_PATH + self.dxf_name)
        print('Contour %s with %d points' % (self.dxf_name, contour.shape[0]))
        return contour

    def get_dxf_size(self):
        assert self.dxf_name is not None
        points = self.load_dxf_points()
        maxx, maxy = max(points[:, 0]), max(points[:, 1])
        return maxx, maxy

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