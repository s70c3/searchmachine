import json
import re
from collections import defaultdict
from shapely.geometry import Polygon, Point

float_regex = r'[-+]?[0-9]*\.?[0-9]*'
float_regex = re.compile(fr'{float_regex}{float_regex}{float_regex}')

def extract_transforms( transform_s):
    # example of transform_s:  'translate(14845.510 608.08935950) rotate(180.0)'
    numbers = re.findall(float_regex, transform_s)
    numbers = list(filter(len, numbers))
    numbers = list(map(float, numbers))
    return numbers


class PackedFigure:
    def __init__(self, etree_obj, json_obj):
        assert int(etree_obj.getAttribute('id')) == json_obj['id']
        self.points = json_obj['points']
        self.id = json_obj['id']
        self.type_id = json_obj['type_id']

        transform = etree_obj.getAttribute('transform')
        self.tr_x, self.tr_y, self.rot = extract_transforms(transform)

    def translate_(self, dx, dy):
        for i, coords in enumerate(self.points):
            self.points[i]['x'] += dx
            self.points[i]['y'] += dy

    def get_pos(self):
        return self.points[0]

    def __repr__(self):
        return f'PackedFigure id={self.id} translate({self.tr_x}, {self.tr_y}) rot({self.rot})'


class Packmap:
    def __init__(self, w, h, etree_obj):
        self.w, self.h = w, h
        self.details = []

    def add(self, detail: PackedFigure):
        self.details.append(detail)

    def get_kim(self):
        square = self.w * self.h
        used_square = sum(list(map(self._get_fig_square, self.details)))
        return used_square / square

    def _get_fig_square(self, fig):
        points = list(map(lambda xy: Point(xy['x'], xy['y']), fig.points))
        poly = Polygon(points)
        return poly.area

    def get_ids_per_list(self):
        types = defaultdict(lambda: 0)
        for detail in self.details:
            types[str(detail.type_id)] += 1
        return dict(types)

    def get_details_info(self):
        info = []
        for detail in self.details:
            # xy = detail.get_pos()
            info.append({'id': detail.type_id,
                         'rotation': detail.rot})
        return info

    @staticmethod
    def get_packmap_size(packmap):
        """Returns width and height of givel xml.dom.minidom object"""
        is_rect = lambda elem: 'Element: rect at' in str(elem)
        elems = list(filter(is_rect, packmap.childNodes))
        rect = elems[0]
        width = float(rect.getAttribute('width'))
        height = float(rect.getAttribute('height'))
        return width, height


def packmap_from_etree_and_json(packmap, jsoned_shapes):
    w, h = Packmap.get_packmap_size(packmap)
    pm = Packmap(w, h, packmap)
    is_g = lambda elem: 'Element: g at' in str(elem)
    figures = list(filter(is_g, packmap.childNodes))
    for elem in figures:
        idx = int(elem.getAttribute('id'))
        jsn = jsoned_shapes[idx - 1]  # already sorted by id
        fig = PackedFigure(elem, jsn)
        pm.add(fig)
    return pm