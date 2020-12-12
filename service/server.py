from sys import setrecursionlimit
import re
import os
import numpy as np
import json
from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.ioloop import IOLoop

# techprocesses and ocr imports
from text_recog import ocr
from predict_norms.api import predict_operations_and_norms
from nomeclature_recognition.api import extract_nomenclature
# price imports
from service.models import PredictModel
from service.logger import LoggerYellot
from service.feature_extractors import DetailData
from service.request_validators import TablularDetailDataValidator, PDFValidator
# packing imports
from packing.models.rect_packing_model.packing import pack_rectangular
from packing.models.poly_packing import pack_polygonal
# from packing.models.neural_packing import pack_neural
from packing.models.request_parsing import RectPackingParameters, DxfPackingParameters

# To pack thousands of details on one list with rectangular packing (based on kd tree),
# recursion  limit should be huge
setrecursionlimit(10**6)

def make_app():
    urls = [('/helth', HelthHandler),
            ('/turnoff', TurnoffHandler),
            ('/calc_detail', CalcDetailByTableHandler),
            ('/calc_detail_schema', CalcDetailBySchemaHandler),
            ('/pack_details', PackDetailsRectangular),
            ('/pack_details_polygonal', PackDetailsPolygonal),
            ('/pack_details_neural', PackDetailsNeural),
            ('/files/(.*)', SendfileHandler, {'path': os.getcwd() + '/service/files/'})]
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


class CalcDetailBySchemaHandler(RequestHandler):
    def _select_parameter(self, predicted_value, value_from_req):
        info = {'predicted': str(predicted_value),
                'given': str(value_from_req),
                'used_value': None}
        if value_from_req is None:
            if predicted_value is None:
                return None, info
            else:
                info['used_predicted'] = True
                return predicted_value, info
        else:
            info['used_given'] = True
            return value_from_req, info

    def _preprocess_sizes(self, value):
        try:
            value = str(value)
            value = sorted(value.replace(',', '.').split('-'))
        except Exception as e:
            print('[err] while parsing size', value, 'error', e, 'occured')
            return None
        return value

    def _predict_linsizes(self, pdf_img):
        linsizes = ocr.extract_sizes(pdf_img)
        return linsizes[1]

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

    def _predict_ops_and_norms(self, pdf_img, material, mass, thickness, length, width):
        ops_norms = predict_operations_and_norms(pdf_img, material, mass, thickness, length, width).result
        return ops_norms

    def _create_broken_pdf_responce(self, price):
        return {'price': price,
                'linsizes': [],
                'techprocesses': [],
                'info': {'predicted_by': [{'tabular': True}, {'scheme': False}],
                         'errors': [
                             {'description': 'Cant predict techprocesses without pdf paper'},
                             {'description': 'Cant predict linear sizes without pdf paper'}
                         ]}
                }

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

        candidate_str = preprocess(material_string)
        candidate = get_best_candidate(candidate_str)
        thickness = fetch_thickness(candidate)
        return thickness

    def _fetch_detail_name(self, detail_name):
        if detail_name is None:
            return None
        try:
            name = detail_name.split()[0].lower()
            return name
        except:
            return None

    def _create_responce(self, price, mass, material, linsizes, techprocesses, thickness):
        return {'price': price,
                'linsizes': linsizes,
                'mass': mass,
                'material': material,
                'techprocesses': techprocesses,
                'material_thickness': thickness,
                'info': {'predicted_by': [{'tabular': True}, {'scheme': True}],
                         'errors': []}
                }

    def _get_prediction_failed_message(self, failed_predict_op, param_name, param_value, used_params):
        return {'prediction_error': f'Cant read valid {param_name} from pdf file. {param_name} was not given in request params. Cant make predict {failed_predict_op} without it',
                f'predicted_{param_name}': str(param_value),
                'used_params': used_params}

    def post(self):
        # parse request data
        used_params = {}
        params_validator = TablularDetailDataValidator()
        parse_errors = params_validator.parse_sizes(self)
        if len(parse_errors):
            self.write({'parse_errors': parse_errors})
            return

        parse_errors = params_validator.get_parse_errors(self)

        # get clean data
        try:
            size_x, size_y, size_z = list(map(lambda s: float(s), self.get_argument('size').split('-')))
            sizes = [size_x, size_y, size_z]
        except:
            self.write(self._get_prediction_failed_message('price', 'size', self.get_argument('size'), used_params))
            return
        used_params['sizes'] = sizes

        # validate attached pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            return

        # predict detail parameters by pdf paper
        img = pdf_validator.get_image()
        mass = params_validator.mass
        pred_mass, pred_material = self._predict_nomenclature(img)
        mass = mass or pred_mass
        try:
            parsed_mass = list(filter(lambda c: c.isdigit() or c == '.', str(mass)))
            parsed_mass = ''.join(parsed_mass)
            mass = abs(float(parsed_mass))
        except ValueError:
            self.write(self._get_prediction_failed_message('price', 'mass', mass, used_params))
            return
        used_params['mass'] = mass

        material = params_validator.material or pred_material
        if material is None:
            self.write(self._get_prediction_failed_message('price', 'material', material, used_params))
            return
        used_params['material'] = material
        
        features = DetailData(size_x, size_y, size_z, mass, material).preprocess()

        linsizes = self._predict_linsizes(img)
        used_params['linsizes'] = 'linsizes'
        price = price_model.predict_price(features, linsizes)

        thickness = self._fetch_material_thickness(material)
        pred_thickness = self._fetch_material_thickness(pred_material)
        if thickness is None:
            if pred_thickness is None:
                self.write(self._get_prediction_failed_message('techprocesses', 'material_thickness', pred_thickness, used_params))
                return
            thickness = pred_thickness
        used_params['material_thickness'] = thickness

        detail_name = self._fetch_detail_name(params_validator.detail_name)
        detail_name = detail_name# or self._fetch_detail_name(material)
        if detail_name is None:
            self.write(self._get_prediction_failed_message('techprocesses', 'detail_name', 'We are not predicting it now', used_params))
            return

        ops_and_norms = self._predict_ops_and_norms(img, detail_name, mass, thickness, length=sorted(sizes)[-1], width=sorted(sizes)[-2])
        if ops_and_norms is None:
            self.write({'prediction_error': 'Error with picture while predicting techprocesses. Maybe no projections found on pdf. Try another pdf', 'used_params': used_params})
            return

        make_obj = lambda op_norm: {'name': op_norm[0], 'norm': op_norm[1]}
        ops_objects = list(map(make_obj, ops_and_norms))

        info = self._create_responce(price, mass, material, linsizes, ops_objects, thickness)
        logger.info('calc_detail', 'ok', {'size': [size_x, size_y, size_z], 'mass': mass, 'material': material})
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
        self.write(errors_or_packing_info)


class PackDetailsPolygonal(RequestHandler):
    """
    Packing of polygon objects. Must include dxfs as a source of polygons
    """

    def post(self):
        params = json.loads(self.request.body.decode('utf-8'))
        params = DxfPackingParameters(params)
        errors_or_packing_info = pack_polygonal(params)
        self.write(errors_or_packing_info)


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
