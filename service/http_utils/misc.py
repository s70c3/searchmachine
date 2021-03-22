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


class TurnoffHandler(RequestHandler):
    def post(self):
        os.system('kill $PPID')