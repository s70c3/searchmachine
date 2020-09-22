import os
from math import log1p, log, sqrt, exp
import numpy as np

from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes

from text_recog import ocr
from text_recog import utils
from predict_operations.predict import pilpaper2operations
from service.logger import LoggerYellot
from service.data import RequestData, DetailData
from service.models import CbRegressor, CbClassifier

from nomeclature_recognition.api import extract_nomenclature


def make_app():
    urls = [('/helth', HelthHandler),
            ('/sendfile', SendfileHandler),
            ('/turnoff', TurnoffHandler),
            ('/calc_detail', CalcDetailByTableHandler),
            ('/calc_detail_schema', CalcDetailBySchemaHandler)]
    return Application(urls)


def predict_tabular(tabular_features):
    price_class = model_price_classifier.predict(tabular_features)
    features = tabular_features + [price_class[0]]
    logprice = model_tabular.predict(features)
    price = round(exp(logprice), 2)
    return price


def predict_tabular_paper(tabular_features, linsizes):
    price_class = model_price_classifier.predict(tabular_features)
    paper_features = utils.fast_hist(linsizes, bins=10)

    features = tabular_features + [price_class[0]] + list(paper_features)
    print('f', features)
    logprice = model_tabular_paper.predict(features)
    price = round(exp(logprice), 2)
    return price


# errcodes
# 1 - not enough arguments in request or invalid argument values
# 2 - invalid detail data format
# 3 - invalid detail data: Unknown material
# 4 - legacy - /sendfile invalid files

class CalcDetailByTableHandler(RequestHandler):
    def post(self):
        # current model columns
        # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category, price_category
        data = RequestData(self)

        if not data.is_valid_table_query():
            logger.error(1, 'calc_detail', 'not enough arguments in request or invalid argument values', data.get_request_data())
            self.write({'code': 1, 'error': 'not enough arguments in request or invalid argument values', 'price': None})
            return

        # get clean data
        size_x, size_y, size_z = list(map(lambda s: float(s), data.size.split(data.sep)))
        mass = float(data.mass)
        material = data.material.split()[0].lower()
        x = DetailData(size_x, size_y, size_z, mass, material)

        resp = calculate_data(data, x)
        logger.info('calc_detail', 'ok', data.get_request_data(additional=resp))
        return self.write(resp)


    
def detect_material(material_string):
    m = material_string.lower()
    for material in 'жесть круг лента лист петля проволока прокат профиль рулон сетка'.split():
        if material in m:
            return material
    return m
    

class CalcDetailBySchemaHandler(RequestHandler):
    def post(self):
        data = RequestData(self)

        if not data.is_valid_scheme_query():
            logger.error(1, 'calc_detail', 'not enough arguments in request or invalid argument values', data.get_request_data())
            self.write({'code': 1, 'error': 'not enough arguments in request or invalid argument values', 'price': None})
            return

        size_x, size_y, size_z = list(map(lambda s: float(s), data.size.split(data.sep)))
        # fetch mass and material
        img = data.get_attached_pdf_img()
        nomenclature_data = extract_nomenclature(np.array(img.convert('L')))
        mass = nomenclature_data['mass']
        if ',' in mass: mass = mass.replace(',', '.')
        mass = float(mass)
        material = nomenclature_data['material']
        material = detect_material(material)
        # create dataclass
        x = DetailData(size_x, size_y, size_z, mass, material)

        resp = calculate_data(data, x)
        logger.info('calc_detail', 'ok', data.get_request_data(additional=resp))
        return self.write(resp)


def calculate_data(data, x):
    info = {}
    cant_open_pdf = False
    linsizes = None
    if data.has_attached_pdf:
        #  paper attached
        # try extract sizes from image
        try:
            img = data.get_attached_pdf_img()
            linsizes = ocr.extract_sizes(img)
            price = predict_tabular_paper(x.preprocess(), linsizes)  # round(exp(model_tabular_paper.predict(x)), 2)
            operations = pilpaper2operations(img)
            nomenclature_data = extract_nomenclature(np.array(img.convert('L')))
            info['nomenclature_data'] = nomenclature_data

        except (PDFPageCountError, TypeError):
            cant_open_pdf = True

    # no paper attached or fallback to tabular prediction
    if not data.has_attached_pdf or cant_open_pdf:
        price = predict_tabular(x.preprocess())  # round(exp(model_tabular.predict(x)), 2)
        info = {'predicted_by': [{'tabular': True}, {'scheme': False}],
                'error': 'Cant predict operations without paper'}
        operations = []
        if cant_open_pdf:
            info['predicted_by'].append({'error': 'Tried read data from pdf. Convertion error occured'})

    resp = {'price': price, "techprocesses": operations, 'info': info, 'used_params': x.get_data_dict(), 'linsizes': linsizes}
    return resp


class HelthHandler(RequestHandler):
    def get(self):
        self.write({'status': 'working'})

        

class TurnoffHandler(RequestHandler):
    def post(self):
        logger.info('turnoff', 'ok')
        os.system('kill $PPID')


        
class SendfileHandler(RequestHandler):
    def post(self):
        if 'paper' not in request.files:
            return jsonify({'error': 'Cant get file from request. Maybe u should use \'paper\'\
                                      as attached file name', 'price': None})
        try:
            # assume that there is only one file attached
            first_key = list(request.files.keys())[0]
            file = request.files[first_key].stream.read() # flask FileStorage object
            img = convert_from_bytes(file)[0]
        except PDFPageCountError:
            logger.error('sendfile', 'failed with files ' + str(list(request.files.keys())))
            return jsonify({'code': 4, 'error': 'Given file is not a pdf file. Internal converting error', 'price': None})

        img.save('./recieved_files/paper.pdf')
        logger.info('sendfile', 'ok', str(list(request.files.keys())))
        return self.write({'ok': 'File saved', 'price': None, 'files': list(request.files)})



if __name__ == "__main__":
    logger = LoggerYellot('./service.log', False)
    model_tabular = CbRegressor('./weights/cbm_tabular_regr.cbm')
    model_tabular_paper = CbRegressor('./weights/cbm_maxdata_regr.cbm')
    model_price_classifier = CbClassifier('./weights/cbm_price_class.cbm')
    print('Models have loaded')

    app = make_app()
    app.listen(5022)
    IOLoop.instance().start()
