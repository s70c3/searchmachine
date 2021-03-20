from tornado.web import RequestHandler

class PackBase(RequestHandler):
    def initialize(self, packing_f):
        self.packing_f = packing_f

    def _make_detail(self, detail, idx):
        raise NotImplementedError

    def _check_fits_material(self, detail_w, detail_h, material_w, material_h):
        return (detail_w <= material_w and detail_h <= material_h) or \
               (detail_w <= material_h and detail_h <= material_w)