import os
from math import log1p, log, sqrt, exp

from flask import Flask, request, jsonify
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes

from text_recog import ocr
from text_recog import utils
from predict_operations.predict import pilpaper2operations
from service.logger import LoggerYellot
from service.data import RequestData
from service.models import CbRegressor, CbClassifier
app = Flask(__name__)



def predict_tabular(tabular_features):
    price_class = model_price_classifier.predict(tabular_features)
    features = tabular_features + [price_class[0]]
    print('f', features)
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

@app.route("/calc_detail/", methods=['POST'])
def calc_price():
    # current model columns
    # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category, price_category
    data = RequestData(request)
    
    params = ('size', 'mass', 'material')
    if not all([item in request.args for item in params]):
        logger.error(1, 'calc_detail', 'not enought arguments in request', data.get_request_data())
        return jsonify({'code': 1, 'error': 'not enough detail parameters in request', 'price': None})

    try:
        x = data.preprocess()
    except ValueError:
        logger.error(2, 'calc_detail', 'invalid detail data format', data.get_request_data())
        return jsonify({'code': 2, 'error': 'invalid detail data format', 'price': None})
    except KeyError:
        logger.error(3, 'calc_detail' 'invalid detail data: Unknown material')
        return jsonify({'code': 3, 'error': 'invalid detail data. Unknown material', 'price': None})

    info = {}
    cant_open_pdf = False
    if len(request.files) != 0:
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
    if len(request.files) == 0 or cant_open_pdf:
        price = predict_tabular(x) #round(exp(model_tabular.predict(x)), 2)
        info = {'predicted_by': [{'tabular': True}, {'scheme': False}],
                'error': 'Cant predict operations without paper'}
        operations = []
        if cant_open_pdf:
            info['predicted_by'].append({'error': 'Tried read data from pdf. Convertion error occured'})

    
    info['given_docs'] = {'names': list(request.files.keys()), 'count': len(request.files)}
    resp = {'price': price, "techprocesses": operations, 'info': info}
    logger.info('calc_detail', 'ok', data.get_request_data(additional={'price': price, 'info': info}))
    return jsonify(resp)


@app.route("/turnoff/", methods=['POST'])
def turnoff():
    logger.info('turnoff', 'ok')
    os.system('kill $PPID')


@app.route("/sendfile/", methods=['POST'])
def handle():
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
    return jsonify({'ok': 'File saved', 'price': None, 'files': list(request.files)})



if __name__ == "__main__":
    logger = LoggerYellot('./service.log')
    model_tabular = CbRegressor('./weights/cbm_tabular_regr.cbm')
    model_tabular_paper = CbRegressor('./weights/cbm_maxdata_regr.cbm')
    model_price_classifier = CbClassifier('./weights/cbm_price_class.cbm')
    
    app.run(debug=False, host='127.0.0.1', port=5022)
