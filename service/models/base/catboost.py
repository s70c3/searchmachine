from catboost import CatBoostRegressor, CatBoostClassifier


class BaseModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model_initialized = False

    def _init_model(self):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError


class CbModel(BaseModel):
    """Base catboost model class"""

    def __init__(self, weights_path):
        super(CbModel, self).__init__(weights_path)

    def _init_model(self, model_cls, *args):
        model = model_cls(args)
        model.load_model(self.weights_path)
        self.model_initialized = True
        return model

    def predict(self, features):
        if not self.model_initialized:
            raise AttributeError('Model not initialized')
        else:
            return self.model.predict(features)


class CbClassifier(CbModel):
    def __init__(self, weights_path):
        super(CbClassifier, self).__init__(weights_path)
        self.model = self._init_model(CatBoostClassifier)


class CbRegressor(CbModel):
    def __init__(self, weights_path, *args):
        super(CbRegressor, self).__init__(weights_path)
        self.model = self._init_model(CatBoostRegressor, args)
