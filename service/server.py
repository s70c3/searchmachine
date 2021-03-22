import os
import sys

sys.path.append('../')  # allow imports in sibling modules

from tornado.web import Application
from tornado.ioloop import IOLoop
from yamlparams.utils import Hparam

from http_utils import SendfileHandler
from http_utils import DxfSizesHandler
from http_utils import DxfSquareHandler
from http_utils import DxfPointsHandler
from http_utils import HelthHandler, TurnoffHandler
from http_utils import CalcDetailByTableHandler, CalcDetailBySchemaHandler
from http_utils import PredictParamsBySchemaHandler
from http_utils.packing import PackRectangular
from http_utils.packing import PackSvgNest
from http_utils.packing import PackHybrid

from models.predict_price import PredictModel
from detail.dxf import Loader

from service.models.packing.rect_packing_model.packing import pack_rectangular
from service.models.packing.svg_nest_packing.packing import SvgNestPacker
from service.models.packing.hybrid_packing.packing import HybridPacker


def make_app(config):
    dxf_params = dict(loader=Loader(with_cache=True, base_path=config.dxf_path))
    urls = [
        ('/health', HelthHandler),
        ('/turnoff', TurnoffHandler),
        ('/calc_detail', CalcDetailByTableHandler, dict(model=price_model)),
        ('/calc_detail_schema', CalcDetailBySchemaHandler, dict(model=price_model)),
        ('/get_params_by_schema', PredictParamsBySchemaHandler),
        
        ('/get_dxf_sizes', DxfSizesHandler, dxf_params),
        ('/dxf/sizes', DxfSizesHandler, dxf_params),
        ('/dxf/square', DxfSquareHandler, dxf_params),
        ('/dxf/points', DxfPointsHandler, dxf_params),
        
        ('/pack_details', PackRectangular, dict(packing_f=pack_rectangular)),
        ('/pack_details_hybrid', PackHybrid, dict(packing_f=HybridPacker())),
        ('/pack_details_svgnest', PackSvgNest, dict(packing_f=SvgNestPacker())),
        ('/pack/rectangular', PackRectangular, dict(packing_f=pack_rectangular)),
        ('/pack/hybrid', PackHybrid, dict(packing_f=HybridPacker())),
        ('/pack/svgnest', PackSvgNest, dict(packing_f=SvgNestPacker())),
        
        # ('/pack_details_polygonal', PackDetailsPolygonal),
        # ('/pack_details_neural', PackDetailsNeural),
        
        ('/files/(.*)', SendfileHandler, {'path': os.getcwd() + config.static_folder})
    ]
    return Application(urls)


if __name__ == '__main__':
    config = Hparam('./config.yml')

    if config.run.models.price:
        price_model = PredictModel(tabular_model_path='cbm_tabular_regr.cbm',
                                   tabular_paper_model_path='cbm_maxdata_regr.cbm',
                                   price_category_model_path='cbm_price_class.cbm')
    else:
        price_model = None
        print("Price model havent loaded due config settings")

    app = make_app(config)
    app.listen(5022)
    print("Ready")
    IOLoop.instance().start()
