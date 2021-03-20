import unittest
import requests as r


class PriceTabularOkTests(unittest.TestCase):
    def _check_default_issues(self, resp):
        self.assertTrue(all(map(lambda key: key in resp, 'price techprocesses info'.split())), resp)
        self.assertTrue(len(resp['techprocesses']) == 0, resp)

    def test_ok_1(self):
        params = {'material': 'Рулон БТ-БШ-О-0,55х1250 08пс-ОН-НР-1',
                  'size': '0.55-475-600',
                  'mass': '0.1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self._check_default_issues(resp)
        self.assertTrue(abs(resp['price'] - 16.13) < 1, resp)

    def test_ok_2(self):
        params = {'size': '4-131-725',
                  'mass': '2.98',
                  'material': 'Лист Б-ПН-О-4х1500х3000 ГОСТ 19903-74 / Ст3сп5 ГОСТ 14637-89'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self._check_default_issues(resp)
        self.assertTrue(abs(resp['price'] - 182.51) < 1, resp)

    def test_ok_3(self):
        params = {'material': "Лист ГОСТ 19903",
                  'size': "550-110-4",
                  'mass': "1.88"}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self._check_default_issues(resp)
        self.assertTrue(abs(resp['price'] - 74.56) < 1, resp)


class PriceTabularBadTests(unittest.TestCase):

    def test_no_material(self):
        params = {'size': '0.55-475-600',
                  'mass': '0.1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)

    def test_no_size(self):
        params = {'material': '',
                  'mass': '0.1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)

    def test_no_mass(self):
        params = {'material': '',
                  'size': '0.55-475-600'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)

    def test_broken_material(self):
        params = {'material': '',
                  'size': '0.55-475-600',
                  'mass': '0.1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)

    def test_broken_size(self):
        params = {'material': 'Рулон БТ-БШ-О-0,55х1250 08пс-ОН-НР-1',
                  'size': '0.55475-600',
                  'mass': '0.1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)

    def test_broken_mass(self):
        params = {'material': 'Рулон БТ-БШ-О-0,55х1250 08пс-ОН-НР-1',
                  'size': '0.55-475-600',
                  'mass': '0,1'}
        resp = r.post(addr+method_tabular, params=params)
        self.assertTrue(resp.status_code == 422, resp)


addr = 'http://127.0.0.1:5022/'
method_tabular = 'calc_detail'