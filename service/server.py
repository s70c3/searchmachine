import os
from math import log1p, log, sqrt, exp

from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes

from text_recog import ocr
from text_recog import utils
from predict_operations.predict import pilpaper2operations
from service.logger import LoggerYellot
from service.data import RequestData
from service.models import CbRegressor, CbClassifier



def make_app():
    urls = [('/helth', HelthHandler),
            ('/sendfile', SendfileHandler),
            ('/turnoff', TurnoffHandler),
            ('/calc_detail', CalcDetailHandler)]
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
# 1 - not enoght arguments in request
# 2 - invalid detail data format
# 3 - invalid detail data: Unknown material
# 4 - legacy - /sendfile invalid files

class CalcDetailHandler(RequestHandler):
    def post(self):
        # current model columns
        # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category, price_category
        data = RequestData(self)

        if not data.is_valid_query():
            logger.error(1, 'calc_detail', 'not enought arguments in request', data.get_request_data())
            self.write({'code': 1, 'error': 'not enough detail parameters in request', 'price': None})
            return
            
        try:
            x = data.preprocess()
        except ValueError:
            logger.error(2, 'calc_detail', 'invalid detail data format', data.get_request_data())
            self.write({'code': 2, 'error': 'invalid detail data format', 'price': None})
            return
        except KeyError:
            logger.error(3, 'calc_detail' 'invalid detail data: Unknown material')
            self.write({'code': 3, 'error': 'invalid detail data. Unknown material', 'price': None})
            return

        info = {}
        cant_open_pdf = False
        if data.has_attached_pdf:
            #  paper attached
            # try extract sizes from image
            try:
                img = data.get_attached_pdf_img()
                linsizes = ocr.extract_sizes(img)
                price = predict_tabular_paper(x, linsizes) #round(exp(model_tabular_paper.predict(x)), 2)
                info = {'predicted_by': [{'tabular': True}, {'scheme': True}]}
                operations = pilpaper2operations(img)
            except PDFPageCountError:
                cant_open_pdf = True

        # no paper attached or fallback to tabular prediction
        if not data.has_attached_pdf or cant_open_pdf:
            price = predict_tabular(x) #round(exp(model_tabular.predict(x)), 2)
            info = {'predicted_by': [{'tabular': True}, {'scheme': False}],
                    'error': 'Cant predict operations without paper'}
            operations = []
            if cant_open_pdf:
                info['predicted_by'].append({'error': 'Tried read data from pdf. Convertion error occured'})


        resp = {'price': price, "techprocesses": operations, 'info': info}
        logger.info('calc_detail', 'ok', data.get_request_data(additional={'price': price, 'info': info}))
        return self.write(resp)



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
