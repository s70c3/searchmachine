from math import log1p, log, sqrt
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor

app = Flask(__name__)


def preprocess_data(request):
    size = request.args.get('size')
    mass = float(request.args.get('mass'))
    material = request.args.get('material')
    sum_ops_time = float(request.args.get('sum_ops_time'))

    mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
    calc_dims = lambda s: list([float(x) for x in s.lower().split('х')])
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
    sum_ops_time = sum_ops_time
    density = mass / mul(calc_dims(size))  # mass / volume
    material_category = materials[get_material(material)]

    return [size2, size3, log_volume, log_mass, sqrt_mass, sum_ops_time, density, material_category]


def load_model(model_path):
    model = CatBoostRegressor()
    model = model.load_model(model_path)
    return model


@app.route("/calc_price/", methods=['POST'])
def calc_price():
    # current model columns
    # ['size2', 'size3', 'log_volume', 'log_mass', 'sqrt_mass', 'sum_time', 'density', 'material_category']

    x = preprocess_data(request)
    pred = model.predict(x)

    return jsonify(pred)


if __name__ == "__main__":
    model = CatBoostRegressor()
    model = model.load_model('weights.cbm')
    app.run()
