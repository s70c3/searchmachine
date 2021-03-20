import requests as r
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes


class DetailValidator:
    @staticmethod
    def check_size(size: str) -> bool:
        is_numeric = lambda s: all(map(lambda c: c.isdigit() or c == '.', s))
        try:
            sep = '-'
            assert size is not None
            assert isinstance(size, str)
            assert len(size.split(sep)) == 3
            assert all(map(is_numeric, size.split(sep)))
            return True
        except:
            return False

    @staticmethod
    def check_mass(mass: float) -> bool:
        return mass > 0

    @staticmethod
    def check_material(material: str) -> bool:
        return len(material) > 0


class PDFValidator:
    _errors = []
    _img = None
    def _create_error_obj(self, msg):
        return {'description': msg}

    def _download(self, pdf_link):
        if pdf_link is None:
            return self._errors.append(self._create_error_obj('pdf link not given'))

        try:
            file = r.get(pdf_link, allow_redirects=True)
            file = file.content
            return file
        except Exception as e:  # todo: specify exception
            return self._errors.append(self._create_error_obj('Cant download pdf file'))

        return file

    def _validate_pdf(self, file):
        try:
            img = convert_from_bytes(file)[0]
            self._img = img
        except PDFPageCountError as e:  # todo: specify exception
            return self._errors.append(self._create_error_obj('Cant convert pdf to image. Maybe broken pdf or not pdf file'))

    def get_parse_errors(self, request):
        # check if link exists and file exists
        link = request.get_argument('pdf_link')
        file = self._download(link)
        if len(self._errors):
            return self._errors

        self._validate_pdf(file)
        return self._errors

    def get_image(self):
        assert self._img is not None
        print('dtype', type(self._img))
        return self._img
