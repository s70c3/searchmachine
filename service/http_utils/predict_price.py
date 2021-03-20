from math import log1p, sqrt, log
from tornado.web import RequestHandler
from webargs import fields
from webargs.tornadoparser import use_args

from service.models.text_recog.model import LinearSizesModel
from service.http_utils.validation import DetailValidator
from .validation import PDFValidator
from service.models.predict_norms.api import predict_operations_and_norms


class CalcDetailHandlerBase(RequestHandler):
    schema = {'size': fields.Str(validate=DetailValidator.check_size, required=True),
              'mass': fields.Number(validate=DetailValidator.check_mass, required=True),
              'material': fields.Str(validate=DetailValidator.check_material, required=True)}

    def initialize(self, model):
        self.model = model

    def _make_catboost_features(self, size_x, size_y, size_z, mass, material):
        # current features
        # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume,
        # log_density, material_category, price_category
        mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]

        material_freqs = {'жесть': 11,
                          'круг': 131,
                          'лента': 75,
                          'лист': 10052,
                          'петля': 38,
                          'проволока': 21,
                          'прокат': 2,
                          'профиль': 3,
                          'рулон': 20906,
                          'сетка': 4}

        def get_material(mat):
            if mat not in material_freqs.keys() or material_freqs[mat] < 70:
                mat = 'too_rare'
            return mat

        volume = size_x * size_y * size_z
        log_volume = log1p(volume)
        log_mass = log1p(mass)
        sqrt_mass = sqrt(mass)
        log_density = log(1000 * mass / volume)  # log mass / volume
        material_category = get_material(material)

        return [size_x, size_y, size_z, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category]

    def _parse_request_size(self, size):
        return list(map(lambda s: float(s), size.split('-')))

    def _parse_request_material(self, material):
        return material.lower().split()[0]

    def _create_response(self, price, techprocesses=[], errors=[]):
        return {'price': price,
                'techprocesses': techprocesses,
                'info': {'errors': errors}
                }


class CalcDetailByTableHandler(CalcDetailHandlerBase):

    @use_args(CalcDetailHandlerBase.schema, location='query')
    def post(self, reqargs):
        # get clean data
        try:
            size_x, size_y, size_z = self._parse_request_size(reqargs['size'])
            mass = reqargs['mass']
            material = self._parse_request_material(reqargs['material'])
        except:
            self.set_status(422)
            return self.write({'error': 'Invalid parameters values'})

        features = self._make_catboost_features(size_x, size_y, size_z, mass, material)
        price = self.model.predict(features)
        no_pdf_errors = [{'description': 'Cant predict techprocesses without pdf paper'},
                         {'description': 'Cant predict linear sizes without pdf paper'}]
        info = self._create_response(price, errors=no_pdf_errors)

        # TODO: logger
        # logger.info('calc_detail', 'ok', {'size': [size_x, size_y, size_z], 'mass': mass, 'material': material})
        return self.write(info)


class CalcDetailBySchemaHandler(CalcDetailHandlerBase):
    schema = {**CalcDetailHandlerBase.schema,
              'pdf_link': fields.URL(required=True),
              'detail_name': fields.Str(required=True),
              'material_thickness': fields.Number(required=True)}

    def _predict_ops_and_norms(self, pdf_img, material, mass, thickness, length, width):
        ops_norms = predict_operations_and_norms(pdf_img, material, mass, thickness, length, width)
        result, error = ops_norms.result, ops_norms.error
        if len(error) > 0:
            return None, error
        return result, None

    def _predict_techprocesses(self, sizes, mass, detail_name, thickness, img):
        sizes = sorted(sizes)

        # calc techprocesses by img
        # ops_norms = predict_operations_and_norms_image_only(img)
        # default_techprocesses, err = ops_norms.result, ops_norms.error

        ops_and_norms, errors = self._predict_ops_and_norms(img, detail_name, mass, thickness,
                                                            length=sizes[-1], width=sizes[-2])
        if ops_and_norms is None:
            ops_and_norms = []
        make_obj = lambda op_norm: {'name': op_norm[0], 'norm': op_norm[1]}
        ops_objects = list(map(make_obj, ops_and_norms))
        return ops_objects

    @use_args(schema, location='querystring')
    def post(self, reqargs):
        print(self.schema)
        print(reqargs)
        # parse pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            self.set_status(422)
            return self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})

        # make features
        img = pdf_validator.get_image()
        size_x, size_y, size_z = self._parse_request_size(reqargs['size'])
        mass = float(reqargs['mass'])
        material = self._parse_request_material(reqargs['material'])
        detail_name = reqargs['detail_name']
        thickness = reqargs['material_thickness']
        linsizes = LinearSizesModel.predict(img)

        # predict price
        features = self._make_catboost_features(size_x, size_y, size_z, mass, material)
        price = self.model.predict(features, linsizes)

        # predict techprocesses
        ops_objects = self._predict_techprocesses([size_x, size_y, size_z], mass, detail_name, thickness, img)

        info = self._create_response(price, ops_objects)
        # logger.info('calc_detail', 'ok', {'params': params, 'info': info})
        return self.write(info)
