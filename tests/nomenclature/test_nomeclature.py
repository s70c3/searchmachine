import unittest
import requests as r
import pdf2image
import cv2
import numpy as np


class NomeclatureOkTests(unittest.TestCase):
    def _pil2cv(self, pil_img):
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)

    def _read_pdf(self, pt):
        img = pdf2image.convert_from_path(pt)[0]
        return self._pil2cv(img)

    def test_ok(self):
        pt = 'https://dev.smartprinting.ru/storage/4447.04.00.028.pdf'
        params = {'pdf_link': pt}
        resp = r.post(addr + method_tabular, params=params)
        self.assertTrue(resp.status_code == 200, resp)
        content = resp.json()
        self.assertEqual(content['mass'], 0.04)
        self.assertEqual(content['detail'], 'уголок')
        self.assertEqual(content['name'], '4447.04.00.028')

addr = 'http://127.0.0.1:5022/'
method_tabular = 'get_params_by_schema'