from math import log1p, log, sqrt, exp
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor

app = Flask(__name__)


def preprocess_data(request):
    sep = '/'
    size = request.form.get('size').lower().replace('х', sep).replace('x', sep)
    mass = float(request.form.get('mass').replace(',', '.'))
    material = request.form.get('material')

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


def load_model(model_path):
    model = CatBoostRegressor()
    model = model.load_model(model_path)
    return model


@app.route("/calc_price/", methods=['POST'])
def calc_price():
    # current model columns
    # ['size2', 'size3', 'log_volume', 'log_mass', 'sqrt_mass', 'sum_time', 'density', 'material_category']

    params = ('size', 'mass', 'material')
    if not all([item in request.form for item in params]):
        return jsonify({'error': 'not enough detail parameters in request', 'price': None})
 
    try:
        x = preprocess_data(request)
    except ValueError:
        return jsonify({'error': 'invalid detail data format', 'price': None})
    except KeyError:
        return jsonify({'error': 'invalid detail data. Unknown material', 'price': None})

    pred = model.predict(x)
    price = round(exp(pred), 2)
    return jsonify({'price': price})


if __name__ == "__main__":
    model = CatBoostRegressor()
    model = model.load_model('weights.cbm')
    app.run(debug=False, host='0.0.0.0')
