from tornado.web import RequestHandler

from webargs.tornadoparser import use_args

from .base import PackBase
from .schemas import RectSchema
from service.models.packing.detail import DetailRect


class PackRectangular(PackBase):
    """
    Packing of rectangular objects. May include dxf for packing maps visualizations
    """
    def _make_detail(self, detail, idx):
        obj = DetailRect(detail['width'],
                         detail['height'],
                         detail['quantity'],
                         idx,
                         detail['dxf'])
        return obj

    @use_args(RectSchema, location='json')
    def post(self, reqargs):
        print('rect params', reqargs)

        details = reqargs['details']
        w, h = reqargs['material']['width'], reqargs['material']['height']
        visualize = reqargs['render_packing_maps']

        if not all(map(lambda d: self._check_fits_material(d['width'], d['height'], w, h), details)):
            self.set_status(409)
            return self.write({'error': "Not all detail fit with given material"})

        details_objs = []
        for idx, detail in enumerate(details):
            details_objs.append(self._make_detail(detail, idx))

        errors_or_packing_info = self.packing_f(details_objs, reqargs['material'], visualize)
        # logger.info('calc_detail', 'ok', errors_or_packing_info)
        # log('rect', json.loads(self.request.body.decode('utf-8')), errors_or_packing_info)

        self.write(errors_or_packing_info)