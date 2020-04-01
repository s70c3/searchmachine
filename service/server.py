import pickle
from random import randint
from math import log1p, log, sqrt, exp
from flask import Flask, request, jsonify
import numpy as np
from catboost import CatBoostRegressor
import torch
import torch.nn as nn


app = Flask(__name__)


def preprocess_data(request):
    sep = '/'
    size = request.args.get('size').lower().replace('х', sep).replace('x', sep)
    mass = float(request.args.get('mass').replace(',', '.'))
    material = request.args.get('material')

    mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
    calc_dims = lambda s: list([float(x) for x in s.split(sep)])
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
    material_category = materials[get_material(material)]

    return [size2, size3, log_volume, log_mass, sqrt_mass, density, material_category]


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
    

def load_model_price(model_path):
    model = CatBoostRegressor()
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


@app.route("/calc_detail/", methods=['POST'])
def calc_price():
    # current model columns
    # ['size2', 'size3', 'log_volume', 'log_mass', 'sqrt_mass', 'sum_time', 'density', 'material_category']

    params = ('size', 'mass', 'material')
    if not all([item in request.args for item in params]):
        return jsonify({'error': 'not enough detail parameters in request', 'price': None})
 
    try:
        x = preprocess_data(request)
    except ValueError:
        return jsonify({'error': 'invalid detail data format', 'price': None})
    except KeyError:
        return jsonify({'error': 'invalid detail data. Unknown material', 'price': None})

    price = round(exp(model_price.predict(x)), 2)
    operations = operations_vector_to_names(model_operations(torch.tensor(x[:-1])))

    return jsonify({'price': price, "techprocesses": operations})


if __name__ == "__main__":
    model_price = load_model_price('./weights.cbm') 
    model_operations = load_model_operations('./weights_detail2operation.pt')
    app.run(debug=False, host='127.0.0.1', port=5022)
