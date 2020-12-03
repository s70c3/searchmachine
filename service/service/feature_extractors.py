from math import log1p, log, sqrt
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
