import numpy as np
import os
from PIL import Image, ImageDraw
from shapely import affinity

from .figure import Figure


def too_far(poly1, poly2):
    def get_xy(spoly_xy):
        return spoly_xy[0][0], spoly_xy[1][0]
    x1, y1 = get_xy(poly1.centroid.xy)
    x2, y2 = get_xy(poly2.centroid.xy)
    w0, h0, w1, h1 = poly1.boundary.bounds
    p1w, p1h = w1-w0, h1-h0
    w0, h0, w1, h1 = poly2.boundary.bounds
    p2w, p2h = w1 - w0, h1 - h0

    return (abs(x1-x2) > p2w+p1w) and (abs(y1-y2) > p2h+p1h)


class Packmap:
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
        self.polygons = []
        
    def _get_packmap_bounds(self, extra_polygon=None):
        polygons = list(self.polygons)
        if extra_polygon:
            polygons += [extra_polygon]
        maxx = int(max([poly.bounds[2] for poly in polygons]))
        maxy = int(max([poly.bounds[3] for poly in polygons]))
        if maxx < 10:
            maxx = self.w
        if maxy < 10:
            maxy = self.h
        return maxx, maxy
    
    def render_packmap_with_poly(self, poly):
        img = Image.new('L', self._get_packmap_bounds(), color=0)
        im1 = ImageDraw.Draw(img)
        
        for p in self.polygons + [poly]:
            im1.polygon(list(map(lambda p: (int(p[0]), int(p[1])), p.get_points())), 
                        fill=128, outline=255)
        return img
    
    def render_packmap(self):
        img = Image.new('L', self._get_packmap_bounds(), color=0)
        im1 = ImageDraw.Draw(img)
        
        for p in self.polygons:
            im1.polygon(list(map(lambda p: (int(p[0]), int(p[1])), p.get_points())), 
                        fill=128, outline=255)
        return img
    
    def render_full_packmap(self):
        img = Image.new('L', (self.w, self.h), color=0)
        im1 = ImageDraw.Draw(img)
        
        for p in self.polygons:
            im1.polygon(list(map(lambda p: (int(p[0]), int(p[1])), p.get_points())), 
                        fill=128, outline=255)
        return img

    def _calc_kim(self, polygon):
        # Assume that given polygon doesnt intersects with other
        packmap_shape = self._get_packmap_bounds(polygon)
        packmap_square = packmap_shape[0] * packmap_shape[1]

        total_square = 0.
        for p in self.polygons + [polygon]:
            total_square += p.area

        return total_square / packmap_square
    
    def _calc_true_kim(self, polygon):
        # Assume that given polygon doesnt intersects with other
        packmap_shape = (self.w, self.h) #self._get_packmap_bounds(polygon)
        packmap_square = packmap_shape[0] * packmap_shape[1]
        
        total_square = 0.
        for p in self.polygons+[polygon]:
            total_square += p.area

        return total_square / packmap_square
    
    def calc_packmap_kim(self):
        packmap_square = self.w*self.h
        used_square = 0.
        for p in self.polygons:
            used_square += p.area
        return used_square / packmap_square
    
    def add_polygon(self, polygon):
        assert isinstance(polygon, Figure)
        self.polygons.append(polygon)
        
    def _is_polygon_inside(self, polygon):
        minx, miny, maxx, maxy = polygon.bounds
        return minx >= 0 and miny >= 0 and maxx <= self.w and maxy <= self.h
    
    
    def get_best_kim_coord(self, polygon):
        # Builds No-Fit-Polygon and measures kim of every position on it. Returns best pos and rect kim
        # WARN given polygon's first point should be the most upper-left
        if len(self.polygons) == 0:
            if self._is_polygon_inside(polygon):
                return (polygon, 1.)
            else:
                # raise Exception('Polygon is too large to pack. Polygon', polygon.bounds, 'Map', (self.w, self.h))
                return (None, None)
        
        scores = []
        for poly in self.polygons:
            for coord in list(poly.get_padded_points()) + list(poly.get_bounds_coords()):
                cur_poly = affinity.translate(polygon.spoly, xoff=coord[0], yoff=coord[1])
                
                # check if polygon fits the packmap
                if not self._is_polygon_inside(cur_poly):
                    continue
                
                # ?? check if current polygon is inside any other polygon
                
                # check intersection with other polygons
                is_intersect = False
                for p in self.polygons:
                    if too_far(p.spoly, cur_poly):
                        continue
                    # if dist(p.center, cur_poly.centroid)
                    # TODO rude intersection check with circles
                    if p.spoly.overlaps(cur_poly) or \
                       p.spoly.equals(cur_poly) or \
                       p.spoly.contains(cur_poly) or \
                       cur_poly.contains(p.spoly):
                        is_intersect = True
                        break
                if is_intersect:
#                     xs, ys = cur_poly.exterior.coords.xy
#                     coords = list(zip(list(xs), list(ys)))
#                     fig = Figure(coords)
#                     render = self.render_packmap_with_poly(fig)
#                     n = len(list(filter(lambda file: file.endswith('.png'), os.listdir('./'))))
#                     render.save(f'./renders/render_{n}.png')
                    continue
                    
#                 assert all(map(lambda p: not cur_poly.almost_equals(p.spoly, decimal=2), self.polygons))
                
                kim = self._calc_kim(cur_poly)
                points = list(cur_poly.exterior.coords)
                scores.append((Figure(points), kim))
             
#         print('scores', scores)
        if len(scores) == 0:
            return (None, None)
        best = sorted(scores, key=lambda elem: -elem[1])[0]
        best = (best[0], self._calc_true_kim(best[0]))
        return best