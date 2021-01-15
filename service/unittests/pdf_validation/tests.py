import unittest
import requests as r

class PDFValidationTests(unittest.TestCase):
    def _is_valid(self, res):
        res = res.json()
        assert isinstance(res, dict)
        assert 'parse_error' not in res.keys()
        return True

    def test_valid_pdf1(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4447.01.00.111.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)

    def test_valid_pdf2(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4465.04.01.062.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)

    def test_valid_pdf3(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4447.01.00.081.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)

    def test_valid_pdf4(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4447.01.00.084.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)

    def test_valid_pdf5(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4447.01.00.104.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)

    def test_valid_pdf6(self):
        pdf_link = 'https://dev.smartprinting.ru/storage/4447.01.00.111.pdf'
        res = r.post(url, params={'pdf_link': pdf_link})
        assert self._is_valid(res)
        

url = "http://0.0.0.0:5022/get_params_by_schema"