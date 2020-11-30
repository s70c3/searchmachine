from catboost import CatBoostRegressor, CatBoostClassifier
import torch
import torch.nn as nn



class BaseModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model_initialized = False
        
        
    def _init_model(self):
        raise NotImplementedError
        
    
    def predict(self, data):
        raise NotImplementedError
        
        
        
class CbModel(BaseModel):
    '''Base catboost model class'''
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
        
        
        
# class TorchClassifier(BaseModel):
#     def __init__(self, weights_path):
#         super(TorchClassifier, self).__init__(weights_path)
        
    
#     def _init_model(self):
#         class DetailsOpsModel(nn.Module):
#             def __init__(self, dim_in, dim_hidden, hidden_layers, dim_out):
#                 super(DetailsOpsModel, self).__init__()

#                 def block(dim_in, dim_out):
#                     return nn.Sequential(nn.Linear(dim_in, dim_out),
#                                          nn.LeakyReLU(0.5),
#                                          nn.Dropout(0.2))

#                 self.fc = nn.Sequential(block(dim_in, dim_hidden),
#                                         *[block(dim_hidden, dim_hidden) for _ in range(hidden_layers)],
#                                         block(dim_hidden, dim_out),
#                                         nn.Linear(dim_out, dim_out),
#                                         nn.Sigmoid())

#             def forward(self, x):
#                 x = self.fc(x)
#                 return x

#         # in hid_size hid_layers out
#         model = DetailsOpsModel(6, 30, 2, 58)
#         #model.load_state_dict(torch.load(model_path))
# #         self.model_initialized = True
#         return model

    
#     def predict(self, features):
#         raise NotImplementedError