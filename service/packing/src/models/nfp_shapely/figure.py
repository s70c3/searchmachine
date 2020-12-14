from shapely import geometry, affinity


class Figure:
    def __init__(self, points, detail=None):
        '''
        @param points:  list[(x, y), ]
        @param detail:  Detail, for packing purposes
        '''
        self.spoly = geometry.Polygon(points)
        self.bounds = self.spoly.bounds
        self.area = self.spoly.area
        self.detail = detail
        
    def get_points(self):
        # print('poiiints', list(self.spoly.exterior.coords[:-1]))
        return list(self.spoly.exterior.coords[:-1]) # last is equal to first
    
    def get_bounds_coords(self):
        w0, h0, w1, h1 = self.spoly.bounds
        strange_margin = 2
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