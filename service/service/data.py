from math import log1p, log, sqrt, exp

from pdf2image.exceptions import PDFPageCountError
from pdf2image import convert_from_bytes


class RequestData:
    def __init__(self, request):
        self.request = request
        self.sep = '-'
        self.size = self.request.get_argument('size', None).lower()
        self.mass = float(self.request.get_argument('mass', None).replace(',', '.'))
        self.material = self.request.get_argument('material', None)
        self.files = list(self.request.request.files)
    
    
    def is_valid_query(self):
        return self.size != None and self.mass != None and self.material != None
    
    
    def preprocess(self):
        mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
        calc_dims = lambda s: sorted(list([float(x) for x in s.split(self.sep)]))
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

        size1, size2, size3 = calc_dims(self.size)
        volume = mul(calc_dims(self.size))
        log_volume = log1p(volume)
        log_mass = log1p(self.mass)
        sqrt_mass = sqrt(self.mass)
        log_density = log(1000*self.mass / mul(calc_dims(self.size)))  #log mass / volume
        material_category = get_material(str(self.material))

        return [size1, size2, size3, volume, self.mass, log_mass, sqrt_mass, log_volume, log_density, material_category] 
    
    
    
    def get_attached_pdf_img(self):
        try:
            # assume that there is only one file attached
            first_key = list(self.request.request.files)[0]
            file = self.request.request.files[first_key][0]['body']
            pickle.dump(file, open('shti.pkl', 'wb'))
            img = convert_from_bytes(file)[0]
            return img
        except PDFPageCountError:
            raise  PDFPageCountError  #)))))))))))))))))))
            
            
    def get_request_data(self, additional=None):
        extract = lambda d: {k:v for k,v in d.items()}
        response = {'args': [self.size,
                             self.mass,
                             self.material],
                    'files': list(self.request.request.files)}
        if additional:
            for key, val in additional.items():
                response[key] = val
    
        return response