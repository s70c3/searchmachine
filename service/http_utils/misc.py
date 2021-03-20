import os
import json
from tornado.web import RequestHandler, StaticFileHandler
from webargs import fields
from webargs.tornadoparser import use_args


class SendfileHandler(StaticFileHandler):
    """Serves static files"""
    def parse_url_path(self, url_path):
        return url_path


class HelthHandler(RequestHandler):
    def get(self):
        self.write({'status': 'working'})


class DxfSizesHandler(RequestHandler):
    """Returns dxf width and height if dxf exists in local DB. Throws file errors"""

    def initialize(self, loader, base_path):
        self.loader = loader
        self.base_path = base_path

    @use_args({'dxf': fields.Str(required=True)}, location='querystring')
    def get(self, reqargs):
        dxf_path = self.base_path + reqargs['dxf']
        try:
            dxf = self.loader.load_optimized_dxf(dxf_path)
            w, h = self.loader.get_sizes(dxf)
        except FileNotFoundError:
            self.set_status(404)
            return self.write({'error': 'no such dxf file found'})
        except Exception as e:
            raise e
            self.set_status(500)
            return self.write({'error': 'undefined error.' + str(e)})

        return self.write({'width': w,
                           'height': h,
                           'points': dxf.tolist()})


class TurnoffHandler(RequestHandler):
    def post(self):
        os.system('kill $PPID')