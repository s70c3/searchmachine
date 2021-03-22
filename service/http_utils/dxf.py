import json
from tornado.web import RequestHandler, StaticFileHandler
from webargs import fields
from webargs.tornadoparser import use_args


class AbstractDxfHandler(RequestHandler):
    def initialize(self, loader):
        self.loader = loader

    def get(self, reqargs, f):
        info = {'result': None,
                'error_description': None,
                'error_code': None}
        try:
            info['result'] = f(reqargs['dxf'])
        except FileNotFoundError:
            info['error_description'] = 'no such dxf file found'
            info['error_code'] = 404
        except Exception as e:
            info['error_description'] = 'undefined error.' + str(e)
            info['error_code'] = 500

        return info


class DxfSizesHandler(AbstractDxfHandler):
    """Returns dxf width and height if dxf exists in local DB. Throws file errors"""

    @use_args({'dxf': fields.Str(required=True)}, location='json')
    def get(self, reqargs):
        res = super().get(reqargs, self.loader.get_sizes)
        if res['error_code'] is not None:
            self.set_status(res['error_code'])
            return self.write(res['error_description'])
        else:
            return self.write({'width': res['result'][0],
                               'height': res['result'][1]})


class DxfSquareHandler(AbstractDxfHandler):

    @use_args({'dxf': fields.Str(required=True)}, location='json')
    def get(self, reqargs):
        res = super().get(reqargs, self.loader.get_square)
        if res['error_code'] is not None:
            self.set_status(res['error_code'])
            return self.write(res['error_description'])
        else:
            return self.write({'square': res['result']})


class DxfPointsHandler(AbstractDxfHandler):

    @use_args({'dxf': fields.Str(required=True)}, location='json')
    def get(self, reqargs):
        res = super().get(reqargs, self.loader.get_points)
        if res['error_code'] is not None:
            self.set_status(res['error_code'])
            return self.write(res['error_description'])
        else:
            return self.write({'points': res['result']})
