from sys import setrecursionlimit
import re
import os
import numpy as np
import json
from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.ioloop import IOLoop

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
        ops_norms = predict_operations_and_norms(pdf_img, material, mass, thickness, length, width)
        result, error = ops_norms.result, ops_norms.error
        if len(error) > 0:
            return None, error
        return result, None

    def _predict_ops_and_norms_img(self, img):
        ops_norms = predict_operations_and_norms_image_only(img)
        if len(ops_norms.error) > 0:
            return None, ops_norms.error
        return ops_norms, None

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

    def _get_prediction_failed_message(self, failed_predict_op, param_name, param_value, used_params, techprocesses):
        return {'prediction_error': f'Cant read valid {param_name} from pdf file. {param_name} was not given in request params. Cant make predict {failed_predict_op} without it',
                f'predicted_{param_name}': str(param_value),
                'used_params': used_params,
                'techprocesses': techprocesses}

    def post(self):
        # parse request data
        used_params = {}

        # parse pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            return
        img = pdf_validator.get_image()

        # predict techprocesses by image
        make_obj = lambda op_norm: {'name': op_norm[0], 'norm': op_norm[1]}
        ops_norms = predict_operations_and_norms_image_only(img)
        default_techprocesses, err = ops_norms.result, ops_norms.error
        if not err:
            default_techprocesses = list(map(make_obj, default_techprocesses))
        else:
            default_techprocesses = []

        # parse sizes
        sizes = SizeParameter(request_value=self.get_argument('size', None))
        sizes, info = sizes.get()
        used_params['size'] = info
        if 'error' in info:
            report = self._get_prediction_failed_message('price', 'size', sizes, used_params, default_techprocesses)
            self.write(report)
            return
        size_x, size_y, size_z = sizes



        # predict detail mass
        pred_mass, pred_material = self._predict_nomenclature(img)
        mass = MassParameter(request_value=self.get_argument('mass', None),
                             predicted_value=pred_mass)
        mass, info = mass.get()
        used_params['mass'] = info
        if 'error' in info:
            report = self._get_prediction_failed_message('price', 'mass', mass, used_params, default_techprocesses)
            self.write(report)
            return

        # predict detail material
        material = MaterialParameter(request_value=self.get_argument('material', None),
                                     predicted_value=pred_material)
        material, info = material.get()
        used_params['material'] = info
        if 'error' in info:
            report = self._get_prediction_failed_message('price', 'material', material, used_params, default_techprocesses)
            self.write(report)
            return

        # predict price
        features = DetailData(size_x, size_y, size_z, mass, material).preprocess()
        linsizes = self._predict_linsizes(img)
        used_params['linsizes'] = 'linsizes'
        price = price_model.predict_price(features, linsizes)


        # predict thickness from material
        thickness = ThicknessParameter(request_value=self.get_argument('material_thickness', None),
                                       predicted_value=self._fetch_material_thickness(material))
        thickness, info = thickness.get()
        used_params['material_thickness'] = info
        if 'error' in info:
            report = self._get_prediction_failed_message('techprocesses', 'material_thickness', thickness, used_params, default_techprocesses)
            self.write(report)
            return

        # get detail name
        detail_name = DetailNameParameter(request_value=self.get_argument('detail_name', None))
        detail_name, info = detail_name.get()
        used_params['detail_name'] = info
        if 'error' in detail_name:
            report = self._get_prediction_failed_message('techprocesses', 'detail_name', detail_name, used_params, default_techprocesses)
            self.write(report)
            return


        ops_and_norms, errors = self._predict_ops_and_norms(img, detail_name, mass, thickness, length=sorted(sizes)[-1], width=sorted(sizes)[-2])
        if ops_and_norms is None:
            self.write({'prediction_error': 'Error with picture while predicting techprocesses. Predict by other params. Predictions are not accurate'+errors,
                        'used_params': used_params,
                        'techprocesses': default_techprocesses},
                       )
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
