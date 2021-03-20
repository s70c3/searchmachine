import copy
import json
import shutil
import xml
import os
from statistics import mean
from service.models.packing.svg_nest_packing.utils import packmap_from_etree_and_json
from service.models.packing.utils.save import save_svgs, rm_svgs
from service.models.packing.utils.errors import PackingError

BASE_PATH = "models/packing/"

class SvgNestPacker:

    def _translate_shape(self, shape, dx):
        for i, point in enumerate(shape['points']):
            shape['points'][i]['x'] = float(shape['points'][i]['x'] + dx)
            shape['points'][i]['y'] = float(shape['points'][i]['y'])
        return shape

    def _rearrange_shapes(self, material_width, shapes):
        """Orders shapes in one horizontal line without intersections. SvgNest needs
           details to be non-touching"""
        ordered_shapes = []
        curr_right_bound = material_width + 50
        for shape in shapes:
            minx, maxx = self._get_shape_xbounds(shape)
            shape = self._translate_shape(shape, curr_right_bound - minx)

            width = maxx - minx
            curr_right_bound += width + 50
            ordered_shapes.append(shape)
        return ordered_shapes

    def _get_shape_xbounds(self, shape):
        """Returns min and max x coordinate of a shape"""
        minx = float('inf')
        maxx = 0
        for p in shape['points']:
            x = p['x']
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
        return minx, maxx

    def _divide_svg_per_packmaps(self, path):
        """Returns n packmaps as a xml.dom.minidom objects"""
        dom = xml.dom.minidom.parse(path)

        gs = dom.childNodes[1].childNodes
        is_g = lambda elem: 'Element: g at' in str(elem)
        gs = list(filter(is_g, gs))
        for i in range(len(gs)):
            gs[i].setAttribute('transform', '(0, 0)')

        return gs

    def __call__(self, details, material, iterations, rotations, render=True):
        assert len(details) > 0
        shapes = []

        idx = 1
        for detail in details:
            w, h = detail.get_size()

            shape = detail.to_nest4j_format()
            for i in range(detail.quantity):
                shape_copy = copy.deepcopy(shape)
                shape_copy['id'] = idx + i
                shape_copy['type_id'] = detail.idx
                shapes.append(shape_copy)
            idx += detail.quantity

        shapes = self._rearrange_shapes(w, shapes)

        shapes_str = str(shapes).replace("'", '"')
        path = BASE_PATH +'files/tmp1.json'
        with open(path, 'w') as f:
            w = int(material['width'])
            h = int(material['height'])
            f.write('{"container": { "width": ' + str(w) + ', "height": ' + str(h) + ' },')
            f.write(' "shapes": ' + shapes_str)
            f.write('}')
        os.system(f"java -cp {BASE_PATH}nest4J.jar UseCase.Main {path} {iterations} {rotations}")
        try:
            shutil.move('res.svg', BASE_PATH + 'files/packing.svg')
        except FileNotFoundError:
            # java app crashed
            raise PackingError

        svgs = self._divide_svg_per_packmaps(BASE_PATH + 'files/packing.svg')
        kims = []
        ids_per_list = []
        for svg in svgs:
            packmap = packmap_from_etree_and_json(svg, shapes)
            kims.append(round(packmap.get_kim(), 2))
            ids_per_list.append(packmap.get_ids_per_list())
        if render:
            archive_path = save_svgs(svgs)
        else:
            pass#rm_svgs()
        return {'results': {'materials': {'n': len(kims)},
                            'kim': {'average': 'not implemented',
                                    'all': kims},
                            'ids_per_list': ids_per_list},
                'filepath': archive_path if render else "Rendering disabled"}
