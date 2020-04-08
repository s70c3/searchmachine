import pickle
from random import randint
import os
from math import log1p, log, sqrt, exp
from flask import Flask, request, jsonify
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import torch
import torch.nn as nn
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_path, convert_from_bytes
from text_recog import ocr
from text_recog import utils

app = Flask(__name__)


def preprocess_data_old(request):
    sep = '-'
    size = request.args.get('size').lower()
    mass = float(request.args.get('mass').replace(',', '.'))
    material = request.args.get('material')

    mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
    calc_dims = lambda s: sorted(list([float(x) for x in s.split(sep)]))
    get_material = lambda s: s.split()[0].lower()
    materials = {'лист': 1,
                 'рулон': 2,
                 'жесть': 3,
                 'лента': 4,
                 'прокат': 5}

    size1, size2, size3 = calc_dims(size)
    log_volume = log1p(mul(calc_dims(size)))
    log_mass = log(mass)
    sqrt_mass = sqrt(mass)
    density = mass / mul(calc_dims(size))  # mass / volume

    if get_material(material) not in materials.keys():
        material = 1
    else:
        material_category = materials[get_material(material)]

    return [size2, size3, log_volume, log_mass, sqrt_mass, density, material_category]


def preprocess_data_new(request):
    sep = '-'
    size = request.args.get('size').lower()
    mass = float(request.args.get('mass').replace(',', '.'))
    material = request.args.get('material')

    mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
    calc_dims = lambda s: sorted(list([float(x) for x in s.split(sep)]))
    get_material = lambda s: s.split()[0].lower()
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
    def get_material(s):
        mat = s.split()[0].lower()
        if mat not in material_freqs.keys() or material_freqs[mat] < 70:
            mat = 'too_rare'
        return mat

    size1, size2, size3 = calc_dims(size)
    volume = mul(calc_dims(size))
    log_volume = log1p(volume)
    log_mass = log1p(mass)
    sqrt_mass = sqrt(mass)
    log_density = log(1000*mass / mul(calc_dims(size)))  #log mass / volume
    material_category = get_material(material)

    return [size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category]


def operations_vector_to_names(operations_vector):
    # load categories names
    operations_names = np.array(pickle.load(open('./categories_names.pkl', 'rb')))
    sigmoid = lambda x: 1/(1 + np.exp(-x))

    # calc 1 class elements
    operations_vector = sigmoid(operations_vector.detach().cpu().numpy())
    #print('opers probs:', list(map(lambda x: round(x, 3), operations_vector)))
    threshold = 0.73
    operations_vector = (operations_vector > threshold).astype(np.int)

    # map to names
    ixs = np.where(operations_vector == 1)[0]
    names = operations_names[ixs][:3]
    # beutify
    names = list(map(lambda n: {'title': n}, names))

    return names


def load_cb_model(model_path, classifier=False, categorical_ixs=[]):
    model = CatBoostRegressor()
    if classifier:
        model = CatBoostClassifier()
    if categorical_ixs != []:
        model = CatBoostRegressor(cat_features=categorical_ixs)
    model = model.load_model(model_path)
    return model


def load_model_operations(model_path):
    class DetailsOpsModel(nn.Module):
        def __init__(self, dim_in, dim_hidden, hidden_layers, dim_out):
            super(DetailsOpsModel, self).__init__()

            def block(dim_in, dim_out):
                return nn.Sequential(nn.Linear(dim_in, dim_out),
                                     nn.LeakyReLU(0.5),
                                     nn.Dropout(0.2))

            self.fc = nn.Sequential(block(dim_in, dim_hidden),
                                    *[block(dim_hidden, dim_hidden) for _ in range(hidden_layers)],
                                    block(dim_hidden, dim_out),
                                    nn.Linear(dim_out, dim_out),
                                    nn.Sigmoid())

        def forward(self, x):
            x = self.fc(x)
            return x

    # in hid_size hid_layers out
    model = DetailsOpsModel(6, 30, 2, 58)
    #model.load_state_dict(torch.load(model_path))

    return model


def try_get_attached_pdf_img():
    try:
        # assume that there is only one file attached
        first_key = list(request.files.keys())[0]
        file = request.files[first_key].stream.read() # flask FileStorage object
        img = convert_from_bytes(file)[0]
        return img
    except PDFPageCountError:
        raise  PDFPageCountError  #)))))))))))))))))))


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


@app.route("/calc_detail/", methods=['POST'])
def calc_price():
    # current model columns
    # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category, price_category

    params = ('size', 'mass', 'material')
    if not all([item in request.args for item in params]):
        return jsonify({'error': 'not enough detail parameters in request', 'price': None})

    try:
        x = preprocess_data_new(request)
    except ValueError:
        return jsonify({'error': 'invalid detail data format', 'price': None})
    except KeyError:
        return jsonify({'error': 'invalid detail data. Unknown material', 'price': None})

    info = {}
    cant_open_pdf = False
    if len(request.files) != 0:
        #  paper attached
        # try extract sizes from image
        try:
            img = try_get_attached_pdf_img()
            linsizes = ocr.extract_sizes(img)
            price = predict_tabular_paper(x, linsizes) #round(exp(model_tabular_paper.predict(x)), 2)
            info = {'predicted_by': [{'tabular': True}, {'scheme': True}]}
        except PDFPageCountError:
            cant_open_pdf = True

    # no paper attached or fallback to tabular prediction
    if len(request.files) == 0 or cant_open_pdf:
        price = predict_tabular(x) #round(exp(model_tabular.predict(x)), 2)
        info = {'predicted_by': [{'tabular': True}, {'scheme': False}]}
        if cant_open_pdf:
            info['predicted_by'].append({'error': 'Tried read data from pdf. Convertion error occured'})

    operations = operations_vector_to_names(model_operations(torch.tensor(preprocess_data_old(request)[:-1])))

    return jsonify({'price': price, "techprocesses": operations, 'info': info})


@app.route("/turnoff/", methods=['POST'])
def turnoff():
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
        return jsonify({'error': 'Given file is not a pdf file. Internal converting error', 'price': None})

    img.save('./recieved_files/paper.pdf')
    return jsonify({'ok': 'File saved', 'price': None})



if __name__ == "__main__":
    model_price_classifier = load_cb_model('./text_recog/cbm_price_class.cbm', classifier=True)
    model_tabular = load_cb_model('./text_recog/cbm_tabular_regr.cbm')
    model_tabular_paper = load_cb_model('./text_recog/cbm_maxdata_regr.cbm')

    model_operations = load_model_operations('./weights_detail2operation.pt')

    app.run(debug=False, host='127.0.0.1', port=5022)
