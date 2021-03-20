from webargs.tornadoparser import use_args

from .base import PackBase
from .schemas import DxfSchema
from service.models.packing.detail import DetailPolygonal
from service.models.packing.utils.errors import PackingError



class PackSvgNest(PackBase):

    def _make_detail(self, detail, idx):
        obj = DetailPolygonal(detail['quantity'],
                              idx,
                              detail['dxf'])
        return obj

    @use_args(DxfSchema, location='json')
    def post(self, reqargs):
        details = reqargs['details']
        material = reqargs['material']
        w, h = material['width'], material['height']

        details_objs = []
        for idx, detail in enumerate(reqargs['details']):
            details_objs.append(self._make_detail(detail, idx))

        if not all(map(lambda d: self._check_fits_material(d.w, d.h, w, h), details_objs)):
            self.set_status(409)
            return self.write({'error': "Not all detail fit with given material"})

        try:
            packing_info = self.packing_f(details_objs, material, reqargs['iterations'], reqargs['rotations'])
            return self.write(packing_info)
        except PackingError:
            self.set_status(500)
            return self.write({'error': "internal packing algorithm error"})
        except Exception as e:
            self.set_status(500)
            self.write({'error': "Unexpected error " + str(e)})
            raise e
