from math import exp, trunc
import numpy as np
from pathlib import Path
from service.models.base import CbClassifier, CbRegressor

WEIGHTS_PATH = 'models/predict_price/weights/'

class PredictModel():
    def __init__(self, tabular_model_path, tabular_paper_model_path, price_category_model_path):
        self.tabular_model = CbRegressor(WEIGHTS_PATH + tabular_model_path)
        self.tabular_paper_model = CbRegressor(WEIGHTS_PATH + tabular_paper_model_path)
        self.price_category_model = CbClassifier(WEIGHTS_PATH + price_category_model_path)

    def _predict_price_category(self, features):
        price_class = self.price_category_model.predict(features)
        return price_class[0]

    def _predict_tabular(self, features):
        price_class = self._predict_price_category(features)
        features = features + [price_class]
        logprice = self.tabular_model.predict(features)
        price = round(exp(logprice), 2) # predictions are in log space
        return price

    def _predict_tabular_paper(self, features, linsizes):
        price_class = self._predict_price_category(features)
        paper_features = fast_hist(linsizes, bins=10)
        features = features + [price_class] + list(paper_features)

        logprice = self.tabular_paper_model.predict(features)
        price = round(exp(logprice), 2) # predictions are in log space
        return price

    def predict(self, features, linsizes=None):
        '''
        Predicts price on given data
        @param features: list of features for model
        @param linsizes: list of linear sizes detected on pdf
        '''
        info = {}
        if linsizes is None:
            price = self._predict_tabular(features)
        else:
            price = self._predict_tabular_paper(features, linsizes)
        return price


def fast_hist(arr, bins):
    histed = [0 for x in range(bins)]
    if len(arr) == 0:
        return histed
    if isinstance(arr, tuple) and all([len(e)==0 for e in arr]):
        return histed

    mx = max(arr)
    mn = min(arr)
    step = (mx-mn)/bins

    if mx == mn:
        norm = 1 / (mx-mn + 1)
    else:
        norm = 1 / (mx-mn)
    nx = bins

    for elem in arr:
        try:
            ix = trunc((elem - mn) * norm * nx)
            if ix == bins:
                ix -=1
            histed[ix] += 1
        except:
            pass#rint('-->','el', elem, 'min max', mn, mx)

    return np.array(histed)