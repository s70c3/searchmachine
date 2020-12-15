from shapely import geometry, affinity


class Figure:
    def __init__(self, points, detail=None, padding=5):
        '''
        @param points:  list[(x, y), ]
        @param detail:  Detail, for packing purposes
        '''
        self.spoly = geometry.Polygon(points)
        self.spoly = self.spoly.simplify(tolerance=8)
        self.spoly = affinity.translate(self.spoly, padding, padding)
        self.bounds = self.spoly.bounds
        self.area = self.spoly.area
        self.detail = detail
        self.padding = padding
        
    def get_points(self):
        return list(self.spoly.exterior.coords[:-1]) # last is equal to first
    
    def get_padded_points(self):
        poly = self.spoly.exterior.parallel_offset(self.padding,
                                                   side='left',
                                                   join_style=3,
                                                   mitre_limit=0.1)
        poly = Figure(list(poly.coords))
        poly_w0, poly_h0, _, _ = poly.spoly.bounds
        poly = poly.move(-poly_w0, -poly_h0)
        coords = poly.get_points()
        return coords

    def get_bounds_coords(self):
        w0, h0, w1, h1 = self.spoly.bounds
        strange_margin = self.padding//2
        return [(w0-strange_margin, h0-strange_margin),
                (w0-strange_margin, h1+strange_margin),
                (w1+strange_margin, h0-strange_margin),
                (w1+strange_margin, h1+strange_margin)]
    
    def get_shape(self):
        minx, miny, maxx, maxy = self.bounds
        w, h = maxx-minx, maxy-miny
        return (w, h)
    
    def move(self, dx, dy=None):
        if dy is None:
            dy = dx
        poly = affinity.translate(self.spoly, xoff=dx, yoff=dy)
        return Figure(list(poly.exterior.coords), self.detail)
    
    def rotate(self, deg):
        poly = affinity.rotate(self.spoly, deg)
        minx, miny, maxx, maxy = poly.bounds
        poly = affinity.translate(poly, xoff=-minx, yoff=-miny)
        return Figure(list(poly.exterior.coords), self.detail)

    def copy(self):
        return Figure(self.get_points(), self.detail)
    
    def __len__(self):
        return len(self.get_points())
    
    def __eq__(self, other):
        assert isinstance(other, Figure)
        return sorted(self.get_points()) == sorted(other.get_points())