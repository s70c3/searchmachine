import unittest
import requests as r

class SizesTest(unittest.TestCase):
    def _check_default(self, resp):
        self.assertTrue('width' in resp, resp)
        self.assertTrue('height' in resp, resp)
        self.assertTrue('points' in resp, resp)

    def test_ok_1(self):
        resp = r.get(addr + method, params={'dxf': '4110.20.021.dxf'})
        self.assertTrue(resp.status_code == 200, resp.json())
        self._check_default(resp.json())

    def test_ok_2(self):
        resp = r.get(addr + method, params={'dxf': '4458.25.16.237.dxf'})
        self.assertTrue(resp.status_code == 200, resp.json())
        self._check_default(resp.json())

    def test_broken_1(self):
        resp = r.get(addr + method, params={'dxf': 'aaa'})
        self.assertTrue(resp.status_code == 404, resp)

    def test_broken_2(self):
        resp = r.get(addr + method)
        self.assertTrue(resp.status_code == 422, resp)


addr = 'http://127.0.0.1:5022/'
method = 'get_dxf_sizes'