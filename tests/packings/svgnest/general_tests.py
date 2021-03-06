import unittest
import json
import requests as r

TESTS_PATH = '../cases/svgnest/'
ADDRESS = 'http://127.0.0.1:5022/'
METHOD = 'pack_details_svgnest'


class PackingTests(unittest.TestCase):
    def _load_testcase(self, n):
        path = f'{TESTS_PATH}{n}.json'
        params = json.load(open(path))
        return params

    def test_svgnest_case1(self):
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(1))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case2(self):
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(2))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case3(self):
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(3))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case4(self):
        # 940
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(4))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case5(self):
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(5))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case6(self):
        # err
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(6))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case6cut(self):
        # no err
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase('6_cut'))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case7(self):
        # err
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(7))
        self.assertTrue(resp.status_code == 200, resp.json())
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case8(self):
        # err
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(8))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)

    def test_svgnest_case9(self):
        # err
        resp = r.post(ADDRESS + METHOD, json=self._load_testcase(9))
        self.assertTrue(resp.status_code == 200, resp)
        resp = resp.json()
        self.assertTrue('results' in resp, resp)