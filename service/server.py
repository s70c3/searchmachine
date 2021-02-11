from sys import setrecursionlimit
import re
import os
import shutil
import copy
import numpy as np
import json
from statistics import mean
from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.ioloop import IOLoop
import xml

# techprocesses and ocr imports
from text_recog import ocr
from predict_norms.api import predict_operations_and_norms, predict_operations_and_norms_image_only
from nomeclature_recognition.api import extract_nomenclature
# price imports
from service.models import PredictModel
from service.logger import LoggerYellot
from service.feature_extractors import DetailData
from service.request_validators import TablularDetailDataValidator, PDFValidator
from service.parameters import SizeParameter, MassParameter, MaterialParameter, ThicknessParameter, DetailNameParameter
# packing imports
from packing.models.utils.save import save_svgs
from packing.models.rect_packing_model.packing import pack_rectangular
from packing.models.poly_packing import pack_polygonal
from packing.models.svg_nest_packing.utils import packmap_from_etree_and_json
# from packing.models.neural_packing import pack_neural
from packing.models.request_parsing import RectPackingParameters, DxfPackingParameters
from datetime import datetime as dt

# To pack thousands of details on one list with rectangular packing (based on kd tree),
# recursion  limit should be huge
setrecursionlimit(10**6)
PACKING_LOG_PATH = './packing/models/packings.log'
def log(method, params, resp):
    time = dt.now().strftime('%d/%m/%y %H:%M:%S')
    with open(PACKING_LOG_PATH, 'a') as f:
        msg = f"[{time}] /{method} | {params} | {resp}\n"
        print(msg)
        f.write(msg)

def make_app():
    urls = [('/health', HelthHandler),
            ('/turnoff', TurnoffHandler),
            ('/calc_detail', CalcDetailByTableHandler),
            ('/calc_detail_schema', CalcDetailBySchemaHandler),
            ('/get_params_by_schema', PredictParamsBySchemaHandler),
            ('/pack_details', PackDetailsRectangular),
            ('/pack_details_polygonal', PackDetailsPolygonal),
            ('/pack_details_neural', PackDetailsNeural),
            ('/pack_details_svgnest', PackDetailsSvgNest),
            ('/files/(.*)', SendfileHandler, {'path': os.getcwd() + '/packing/models/files/'})]
    return Application(urls)


class CalcDetailByTableHandler(RequestHandler):
    """
    Calculates price with tabular data: sizes, mass and material
    """

    def _create_responce(self, price):
        return {'price': price,
                'linsizes': [],
                'techprocesses': [],
                'info': {'predicted_by': [{'tabular': True}, {'scheme': False}],
                         'errors': [
                             {'description': 'Cant predict techprocesses without pdf paper'},
                             {'description': 'Cant predict linear sizes without pdf paper'}
                         ]}
                }

    def post(self):
        # parse request data
        validator = TablularDetailDataValidator()
        parse_errors = validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_errors': parse_errors})

        # get clean data
        size_x, size_y, size_z = list(map(lambda s: float(s), self.get_argument('size').split('-')))
        mass = float(self.get_argument('mass').replace(',', '.'))
        material = self.get_argument('material').lower().split()[0]
        features = DetailData(size_x, size_y, size_z, mass, material).preprocess()

        price = price_model.predict_price(features)
        info = self._create_responce(price)
        logger.info('calc_detail', 'ok', {'size': [size_x, size_y, size_z], 'mass': mass, 'material': material})
        return self.write(info)


class PredictParamsBySchemaHandler(RequestHandler):
    def _fetch_material_thickness(self, material_string):
        # Fetches thickness of material from the given string with material data
        # If cant fetch anything, returns None
        def preprocess(material):
            # Delete all external symbols
            allowed_symbs = '-xх,. '
            is_allowed = lambda c: c.isdigit() or c in allowed_symbs
            material = material.lower()

            filtred_material = ''
            prev_added_c = None
            for i, c in enumerate(material[1:], start=1):
                if is_allowed(c):
                    if prev_added_c is None or c.isdigit():
                        filtred_material += c
                    elif c in allowed_symbs and c != prev_added_c:
                        filtred_material += c
                    else:
                        continue
                    prev_added_c = c
            return filtred_material

        def get_best_candidate(candidate_sizes_string):
            # Select substring that looks the most closely to sizes
            candidates = candidate_sizes_string.split()
            for s in candidates:
                if ',' in s:
                    return s
            for s in candidates:
                if re.search(r"\d{1,2}(x|х)\d{1,5}(x|х)\d{1,5}", s):
                    return s
            for s in candidates:
                if re.search(r"\d{1,2}(x|х)\d{1,5}", s):
                    return s
            return None

        def fetch_thickness(candidate_str):
            # Fetch thickness string from candidate string
            if candidate_str is None:
                return None

            s = candidate_str.strip('- ').replace(',', '.')
            s = s.replace('x', '@').replace('х', '@').replace('-', '@')
            if '@' in s:
                s = s[:s.index('@')]
            try:
                if float(s) > 100:
                    return None
                else:
                    return float(s)
            except:
                # If doesnt cast or is too large for thickness
                pass
            return None

        if material_string is None:
            return None
        candidate_str = preprocess(material_string)
        candidate = get_best_candidate(candidate_str)
        thickness = fetch_thickness(candidate)
        return thickness

    def _predict_nomenclature(self, pdf_img):

        def detect_material(material_string):
            m = material_string.lower()
            for material in 'жесть круг лента лист петля проволока прокат профиль рулон сетка'.split():
                if material in m:
                    return material
            return m

        nomenclature_data = extract_nomenclature(np.array(pdf_img.convert('L')))
        print('mass detected', nomenclature_data)
        mass = nomenclature_data['mass']
        if ',' in mass:
            mass = mass.replace(',', '.')
        material = nomenclature_data['material']
        try:
            material = detect_material(material)
        except AttributeError: # if material is None
            pass
        return mass, material

    def post(self):
        # validate pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            return
        img = pdf_validator.get_image()
        given_material = self.get_argument('material', None)

        pred_mass, pred_material = self._predict_nomenclature(img)
        pred_thickness = self._fetch_material_thickness(pred_material)
        req_material_thickness = self._fetch_material_thickness(given_material)

        params = {'mass': None,
                  'material': None,
                  'material_thickness_by_img': None,
                  'meterial_thickness_by_given_material': None}
        params_keys = 'mass material material_thickness_by_img meterial_thickness_by_given_material'.split()
        params_classes = [MassParameter, MaterialParameter, ThicknessParameter, ThicknessParameter]
        params_predicted = [pred_mass, pred_material, pred_thickness, req_material_thickness]
        for key, val, Cls in zip(params_keys, params_predicted, params_classes):
            val = Cls(predicted_value=val)
            val, info = val.get()
            if 'error' not in info:
                params[key] = val

        return self.write(params)


class CalcDetailBySchemaHandler(RequestHandler):

    def _predict_ops_and_norms(self, pdf_img, material, mass, thickness, length, width):
        ops_norms = predict_operations_and_norms(pdf_img, material, mass, thickness, length, width)
        result, error = ops_norms.result, ops_norms.error
        if len(error) > 0:
            return None, error
        return result, None

    def _predict_linsizes(self, pdf_img):
        linsizes = ocr.extract_sizes(pdf_img)
        return linsizes[1]

    def _create_responce(self, price, params, techprocesses, errors, warnings):
        res =  {'price': price,
                'techprocesses': techprocesses,
                'info': {'warnings': warnings, 'errors': errors},
                'params': params
                }
        for key, val in params.items():
            res['params'][key] = val
        return res

    def _fetch_request_params(self):
        # Fetches, validates and parses paramteres below from request
        # Returns filled dict from below
        params = {'size_x': None, 'size_y': None, 'size_z': None,
                  'mass': None,
                  'material': None,
                  'material_thickness': None,
                  'detail_name': None}

        # size
        sizes = SizeParameter(request_value=self.get_argument('size', None))
        sizes, info = sizes.get()
        if 'error' not in info:
            params['size_x'], params['size_y'], params['size_z'] = sizes

        params_keys = 'mass material material_thickness detail_name'.split()
        params_classes = [MassParameter, MaterialParameter, ThicknessParameter, DetailNameParameter]
        for key, Cls in zip(params_keys, params_classes):
            val = Cls(request_value=self.get_argument(key, None))
            val, info = val.get()
            if 'error' not in info:
                params[key] = val

        # params_dict = ParametersDict(is_primary=True, **params)
        return params

    def _predict_price(self, params, img):
        # If not enough params, return None
        errors = []
        size_x, size_y, size_z, mass, material = [params[key] for key in 'size_x size_y size_z mass material'.split()]
        if any(list(map(lambda val: val is None, [size_x, size_y, size_z, mass, material]))):
            errors.append({'parameters_error': 'not enough parameters for price prediction'})

        linsizes = self._predict_linsizes(img)
        features = DetailData(size_x, size_y, size_z, mass, material).preprocess()
        price = price_model.predict_price(features, linsizes)
        return price, errors

    def _predict_techprocesses(self, params, img):
        errors = []
        warnings = []

        detail_name = params['detail_name']
        thickness = params['material_thickness']
        sizes = sorted([params['size_x'], params['size_y'], params['size_z']])
        ops_and_norms = None

        ## calc techprocesses by img
        ops_norms = predict_operations_and_norms_image_only(img)
        default_techprocesses, err = ops_norms.result, ops_norms.error

        if (thickness is not None) and (detail_name is not None):
            ops_and_norms, errors = self._predict_ops_and_norms(img, detail_name, params['mass'], thickness,
                                                                length=sizes[-1], width=sizes[-2])
        else:
            warnings.append({'parameters_warning': 'Provide material_thickness and/or detail_name parameters for accurately calculating of techprocesses'})
            # May be None
            ops_and_norms = default_techprocesses

        if ops_and_norms is None:
            ops_and_norms = []

        make_obj = lambda op_norm: {'name': op_norm[0], 'norm': op_norm[1]}
        ops_objects = list(map(make_obj, ops_and_norms))
        return ops_objects, warnings

    def post(self):
        errors = []
        warnings = []
        params = self._fetch_request_params()

        # parse pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            errors.append({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            resp = self._create_responce(None, {}, params, errors, warnings)
            return self.write(resp)

        img = pdf_validator.get_image()

        # predict price
        price, price_errors = self._predict_price(params, img)
        errors += price_errors

        # predict techprocesses
        ops_objects, techprocesses_warnings = self._predict_techprocesses(params, img)
        warnings += techprocesses_warnings

        info = self._create_responce(price, params, ops_objects, errors, warnings)
        logger.info('calc_detail', 'ok', {'params': params, 'info': info})
        return self.write(info)


class HelthHandler(RequestHandler):
    def get(self):
        self.write({'status': 'working'})


class TurnoffHandler(RequestHandler):
    def post(self):
        logger.info('turnoff', 'ok')
        os.system('kill $PPID')


class SendfileHandler(StaticFileHandler):
    def parse_url_path(self, url_path):
        return url_path


class PackDetailsRectangular(RequestHandler):
    """
    Packing of rectangular objects. May include dxf for packing maps visualizations
    """

    def post(self):
        try:
            params = json.loads(self.request.body.decode('utf-8'))
        except json.JSONDecodeError:
            self.write({'error': 'Cant decode given json data'})
        print('rect params', params)
        params = RectPackingParameters(params)
        errors_or_packing_info = pack_rectangular(params)
        logger.info('calc_detail', 'ok', errors_or_packing_info)
        log('rect', json.loads(self.request.body.decode('utf-8')), errors_or_packing_info)

        self.write(errors_or_packing_info)


class PackDetailsPolygonal(RequestHandler):
    """
    Packing of polygon objects. Must include dxfs as a source of polygons
    """

    def post(self):
        params = json.loads(self.request.body.decode('utf-8'))
        params = DxfPackingParameters(params)
        errors_or_packing_info = pack_polygonal(params)
        log('poly', json.loads(self.request.body.decode('utf-8')), errors_or_packing_info)
        self.write(errors_or_packing_info)


class PackDetailsSvgNest(RequestHandler):
    """
    Packing of polygon objects. Must include dxfs as a source of polygons
    """

    def post(self):
        params = json.loads(self.request.body.decode('utf-8'))
        try:
            iterations = int(params['iterations'])
        except:
            return self.write({'errors': ['no iterations argument provided']})
        try:
            rotations = int(params['rotations'])
        except:
            return self.write({'errors': ['no rotations argument provided']})
        params = DxfPackingParameters(params)
        if len(params.errors) > 0:
            return {'errors': params.errors, 'warnings': params.warnings}

        # TODO add ids to shapes
        shapes = []
        idx = 1
        for type_id, detail in enumerate(params.details):
            shape = self.load_shape(detail)
            for i in range(detail.quantity):
                shape_copy = copy.deepcopy(shape)
                shape_copy['id'] = idx + i
                shape_copy['type_id'] = type_id
                shapes.append(shape_copy)
            idx += detail.quantity


        shapes = self.prepare_shapes(params.material_width, shapes)

        shapes = str(shapes).replace("'", '"')
        path = 'packing/models/files/tmp1.json'
        with open(path, 'w') as f:
            w = int(params.material_width)
            h = int(params.material_height)
            f.write('{"container": { "width": '+str(w)+', "height": '+str(h)+' },')
            f.write(' "shapes": ' + shapes)
            f.write('}')
        os.system(f"java -cp packing/models/nest4J.jar UseCase.Main {path} {iterations} {rotations}")
        shutil.move('res.svg', 'packing/models/files/packing.svg')

        svgs = self._divide_svg_per_packmaps('packing/models/files/packing.svg')
        shapes_data = json.load(open(path))['shapes']
        kims = []
        ids_per_list = []
        for svg in svgs:
            packmap = packmap_from_etree_and_json(svg, shapes_data)
            kims.append(round(packmap.get_kim(), 2))
            ids_per_list.append(packmap.get_ids_per_list())
        archive_path = save_svgs(svgs)
        self.write({'results': {'materials': {'n': len(kims)},
                                'kim': {'average': mean(kims),
                                        'all': kims},
                                'ids_per_list': ids_per_list},
                    'filepath': archive_path})

    def load_shape(self, detail):
        points_np = detail.load_dxf_points()
        points = points_np.tolist()
        shape = [{'x': x, 'y': y} for (x,y) in points]
        return {'points': shape, 'id': None}

    def prepare_shapes(self, material_width, shapes):
        ordered_shapes = []
        curr_right_bound = material_width + 50
        for shape in shapes:
            minx, maxx = self._get_shape_xbounds(shape)
            shape = self._translate_shape(shape, curr_right_bound - minx)

            width = maxx - minx
            curr_right_bound += width + 50
            ordered_shapes.append(shape)
        return ordered_shapes

    def _divide_svg_per_packmaps(self, path):
        """Returns n packmaps as a xml.dom.minidom objects"""
        dom = xml.dom.minidom.parse(path)

        gs = dom.childNodes[1].childNodes
        is_g = lambda elem: 'Element: g at' in str(elem)
        gs = list(filter(is_g, gs))
        for i in range(len(gs)):
            gs[i].setAttribute('transform', '(0, 0)')

        return gs


    def _get_shape_xbounds(self, shape):
        minx = float('inf')
        maxx = 0
        for p in shape['points']:
            x = p['x']
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
        return minx, maxx

    def _translate_shape(self, shape, dx):
        for i, point in enumerate(shape['points']):
            shape['points'][i]['x'] = float(shape['points'][i]['x'] + dx)
            shape['points'][i]['y'] = float(shape['points'][i]['y'])
        return shape

class PackDetailsNeural(RequestHandler):
    def post(self):
        self.write({'errors': ['not yet developed']})


if __name__ == "__main__":
    logger = LoggerYellot('./service.log', False)
    price_model = PredictModel(tabular_model_path='./weights/cbm_tabular_regr.cbm',
                               tabular_paper_model_path='./weights/cbm_maxdata_regr.cbm',
                               price_category_model_path='./weights/cbm_price_class.cbm')
    print('Models have loaded')

    app = make_app()
    app.listen(5022)
    IOLoop.instance().start()
