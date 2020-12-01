from math import log1p, log, sqrt, exp
import requests as r
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes
from dataclasses import dataclass

@dataclass
class DetailData:
    size_x: int
    size_y: int
    size_z: int
    mass:   float
    material: str

    def preprocess(self):
        # current features
        # size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category, price_category
        mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]

        get_material = lambda s: s.split()[0].lower()
        material_freqs = {'жесть': 11,
                          'круг': 131,
                          'лента': 75,
                          'лист': 10052,
                          'петля': 38,
                          'проволока': 21,
                          'прокат': 2,
                          'профиль': 3,
                          'рулон': 20906,
                          'сетка': 4}

        def get_material(s):
            mat = s.split()[0].lower()
            if mat not in material_freqs.keys() or material_freqs[mat] < 70:
                mat = 'too_rare'
            return mat

        volume = self.size_x*self.size_y*self.size_z
        log_volume = log1p(volume)
        log_mass = log1p(self.mass)
        sqrt_mass = sqrt(self.mass)
        log_density = log(1000 * self.mass / volume)  # log mass / volume
        material_category = get_material(str(self.material))

        return [self.size_x, self.size_y, self.size_z, volume, self.mass, log_mass, sqrt_mass, log_volume, log_density,
                material_category]

    def get_data_dict(self):
        return {'x': self.size_x, 'y': self.size_y, 'z': self.size_z, 'mass': self.mass, 'material': self.material}

    def get_data(self):
        return [self.size_x, self.size_y, self.size_z, self.mass, self.material]


# class RequestData:
#     def __init__(self, request):
#         self.request = request
#         self.sep = '-'
#         self.size = self.request.get_argument('size', None).lower()
#         self.mass = self.request.get_argument('mass', None)
#         if self.mass is not None:
#             self.mass = float(self.mass.replace(',', '.'))
#         self.material = self.request.get_argument('material', None)
#         self.pdf_link = self.request.get_argument('pdf_link', None)
#         self.has_attached_pdf = self.pdf_link or len(self.request.request.files) or None
#
#     def _is_valid_size(self):
#         # size example  "24.4-13-45"
#         size = self.size
#         if size is None: return False
#         if type(size) != str: return False
#         if size.count(self.sep) != 2: return False
#         if len(
#                 list(filter(lambda s: len(s) > 0,
#                             size.split(self.sep)))) != 3:
#             return False
#         for s in size.split(self.sep):
#             try:
#                 float(s)
#             except:
#                 return False
#         return True
#
#     def _is_valid_mass(self):
#         try:
#             float(self.mass)
#             return True
#         except:
#             return False
#
#     def _is_valid_material(self):
#         return type(self.material) == str and len(self.material) > 0
#
#     def is_valid_table_query(self):
#         return self.size is not None and \
#                self.mass is not None and \
#                self.material is not None and \
#                self._is_valid_size() and \
#                self._is_valid_mass() and \
#                self._is_valid_material()
#
#     def is_valid_scheme_query(self):
#         return self.has_attached_pdf and \
#                self.size is not None and \
#                self._is_valid_size()
#
#     def get_pdf_by_link(self):
#         pdf_link = self.request.get_argument('pdf_link', None)
#         if pdf_link == None:
#             return pdf_link
#
#         try:
#             file = r.get(pdf_link, allow_redirects=True)
#             file = file.content
#             return file
#         except Exception as e:  # todo: specify exception
#             raise e
#
#     # def get_pdf_by_file(self):
#     #     files = self.request.request.files
#     #     if len(list(files)) == 0:
#     #         return None
#     #
#     #     first_key = list(files)[0]
#     #     file = files[first_key][0]['body']
#     #     return file
#
#     def get_attached_pdf_img(self):
#         pdf = self.get_pdf_by_link()
#         if not pdf:
#             self.has_attached_pdf = False
#             return
#
#         try:
#             img = convert_from_bytes(pdf)[0]
#         except PDFPageCountError as e:  # todo: specify exception
#             raise e
#
#         return img
#
#     def get_request_data(self, additional=None):
#         extract = lambda d: {k:v for k,v in d.items()}
#         response = {'args': [self.size,
#                              self.mass,
#                              self.material,
#                              self.pdf_link],
#                     'args?': dir(self.request),
#                    'files': list(self.request.request.files)}
#         if additional:
#             for key, val in additional.items():
#                 response[key] = val
#
#         return response