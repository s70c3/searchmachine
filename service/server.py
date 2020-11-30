import os
import numpy as np

from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.ioloop import IOLoop

from text_recog import ocr
from predict_operations.predict import pilpaper2operations
from service.models import PredictModel
from service.logger import LoggerYellot
from service.feature_extractors import DetailData
from service.request_validators import TablularDetailDataValidator, PDFValidator
from nomeclature_recognition.api import extract_nomenclature
from predict_norms.api import predict_norms, predict_operations


def make_app():
    urls = [('/helth', HelthHandler),
            ('/turnoff', TurnoffHandler),
            ('/calc_detail', CalcDetailByTableHandler),
            ('/calc_detail_schema', CalcDetailBySchemaHandler),
            ('/files/(.*)', SendfileHandler, {'path': os.getcwd()+'/service/files/'})]
    return Application(urls)


class CalcDetailByTableHandler(RequestHandler):
    '''
    Calculates price with tabular data: sizes, mass and material
    '''
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
        mass = nomenclature_data['mass']
        if ',' in mass:
            mass = mass.replace(',', '.')
        mass = float(mass)
        material = nomenclature_data['material']
        material = detect_material(material)
        return mass, material

    def _predict_operations(self, pdf_img):
        operations = pilpaper2operations(pdf_img)
        return operations

    def _predict_norms(self, pdf_img, material, mass, thickness=6):
        # TODO fetch thickness
        norms = predict_norms(pdf_img, material, mass, thickness)
        ops = predict_operations(material, mass, thickness)
        if ops and norms and len(norms) != len(ops):
            print('[WARN] norms and ops from new api has different length')
            ops = None
        return norms, ops

    def _create_broken_pdf_responce(self, price):
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

    def _create_responce(self, price, linsizes, techprocesses):
            return {'price': price,
                    'linsizes': linsizes,
                    'techprocesses': techprocesses,
                    'info': {'predicted_by': [{'tabular': True}, {'scheme': True}],
                             'errors': []}
                    }

    def post(self):
        # parse request data
        validator = TablularDetailDataValidator()
        parse_errors = validator.parse_sizes(self)
        if len(parse_errors):
            self.write({'parse_errors': parse_errors})
            return

        # get clean data
        size_x, size_y, size_z = list(map(lambda s: float(s), self.get_argument('size').split('-')))

        # validate attached pdf
        validator = PDFValidator()
        parse_errors = validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            return

        # predict detail parameters by pdf paper
        img = validator.get_image()
        mass, material = self._predict_nomenclature(img)
        features = DetailData(size_x, size_y, size_z, mass, material).preprocess()

        linsizes = self._predict_linsizes(img)
        price = price_model.predict_price(features, linsizes)
        operations = self._predict_operations(img)
        # Это самый ужасный костыль в моей жизни
        norms, ops = self._predict_norms(img, material, mass, )
        if ops is None:
            ops = operations

        print('operations', operations)
        print('ops', ops)
        print('norms', norms)

        ops_objects = []
        iterable = ops or norms
        if iterable is not None:
            for i in range(len(iterable)):
                if ops is None or i >= len(ops):
                    op = None
                else:
                    op = ops[i]
                if norms is None or i >= len(ops):
                    norm = None
                else:
                    norm = norms[i]
                obj = {'name': op, 'norm': norm}
                ops_objects.append(obj)

        info = self._create_responce(price, linsizes, ops_objects)
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


if __name__ == "__main__":
    logger = LoggerYellot('./service.log', False)
    price_model = PredictModel(tabular_model_path='./weights/cbm_tabular_regr.cbm',
                               tabular_paper_model_path= './weights/cbm_maxdata_regr.cbm',
                               price_category_model_path='./weights/cbm_price_class.cbm')
    print('Models have loaded')

    app = make_app()
    app.listen(5022)
    IOLoop.instance().start()
