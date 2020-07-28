from math import log1p, log, sqrt, exp

from flask import Flask, request, jsonify
from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes



class RequestData:
    def __init__(self, request):
        self.request = request
    
    
    def preprocess(self):
        sep = '-'
        size = self.request.args.get('size').lower()
        mass = float(self.request.args.get('mass').replace(',', '.'))
        material = self.request.args.get('material')

        mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
        calc_dims = lambda s: sorted(list([float(x) for x in s.split(sep)]))
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

        size1, size2, size3 = calc_dims(size)
        volume = mul(calc_dims(size))
        log_volume = log1p(volume)
        log_mass = log1p(mass)
        sqrt_mass = sqrt(mass)
        log_density = log(1000*mass / mul(calc_dims(size)))  #log mass / volume
        material_category = get_material(material)

        return [size1, size2, size3, volume, mass, log_mass, sqrt_mass, log_volume, log_density, material_category] 
    
    
    def get_attached_pdf_img(self):
        try:
            # assume that there is only one file attached
            first_key = list(self.request.files.keys())[0]
            file = self.request.files[first_key].stream.read() # flask FileStorage object
            img = convert_from_bytes(file)[0]
            return img
        except PDFPageCountError:
            raise  PDFPageCountError  #)))))))))))))))))))
            
            
    def get_request_data(self, additional=None):
        extract = lambda d: {k:v for k,v in d.items()}
        response = {'form': extract(self.request.form),
                    'args': extract(self.request.args),
                    'files': extract(self.request.files)}
        if additional:
            for key, val in additional.items():
                response[key] = val

        return response