from abc import ABC, abstractmethod
import requests as r
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes

class Validator(ABC):
    @abstractmethod
    def get_parse_errors(self, request):
        pass


class TablularDetailDataValidator(Validator):
    is_num = lambda _, s: all(map(lambda c: c.isdigit() or c in ',.', s))
    def to_num(self, s):
        s = s.replace(',', '.')
        return abs(float(s))
    _errors = []

    def _check_size(self, size):
        sep = '-'
        assert size is not None
        assert isinstance(size, str)
        assert len(size.split(sep)) == 3
        assert all(map(self.is_num, size.split(sep)))

    def _check_mass(self, mass):
        assert mass is not None
        assert isinstance(mass, str)
        assert all(map(self.is_num, mass))

    def _check_material(self, material):
        assert material is not None
        assert isinstance(material, str)

    def _create_error_obj(self, param_name, param_value, msg):
        text = f'error while reading param {param_name}. {msg}. Found {param_value}'
        return {'description': text}

    def parse_sizes(self, request):
        try:
            size = request.get_argument('size', None)
            self._check_size(size)
            self.sizes = [self.to_num(s) for s in size.split('-')]
        except AssertionError:
            msg = 'Should be string with 3 numbers separated by symbol "-"'
            msg = self._create_error_obj('size', size, msg)
            self._errors.append(msg)
        return self._errors

    def get_parse_errors(self, request):
        self.parse_sizes(request)
        self.sizes = None if self.sizes is None else '-'.join([str(s) for s in self.sizes])
        self.mass = request.get_argument('mass', None)
        self.material = request.get_argument('material', None)
        self.detail_name = request.get_argument('detail_name', None)
        self.material_thickness = request.get_argument('material_thickness', None)

        self.parse_sizes(request)

        try:
            self._check_mass(self.mass)
            abs(float(self.mass.replace(',', '.')))
        except AssertionError:
            msg = 'Should be non-negative float number'
            msg = self._create_error_obj('mass', self.mass, msg)
            self._errors.append(msg)
            
        try:
            self._check_material(self.material)
        except AssertionError:
            msg = 'Should be string with material data'
            msg = self._create_error_obj('material', self.material, msg)
            self._errors.append(msg)

        try:
            self.material_thickness = float(self.material_thickness)
        except: pass
        try:
            self.detail_name = request.get_argument('detail_name', None)
        except: pass

        return self._errors

        
class PDFValidator(Validator):
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
