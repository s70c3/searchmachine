
is_num = lambda s: all(map(lambda c: c.isdigit() or c in ',.', str(s)))
def to_num(s):
    s = s.replace(',', '.')
    return abs(float(s))

class ParameterI:
    def __init__(self, request_value=None, predicted_value=None):
        self.request_value = None
        self.orig_request_value = request_value
        self.orig_predicted_value = predicted_value
        if self.is_valid(request_value):
            self.request_value = self._parse(request_value)

        self.predicted_value = None
        if self.is_valid(predicted_value):
            self.predicted_value = self._parse(predicted_value)



    def is_valid(self, value):
        raise NotImplemented

    def _parse(self, value):
        raise NotImplemented

    def _report(self):
        info = {'values': {'request': self.orig_request_value,
                           'predicted': self.orig_predicted_value},
                'is_valid': {'request': self.is_valid(self.orig_request_value),
                             'predicted': self.is_valid(self.orig_predicted_value)},
                'is_used': {'request': self.request_value is not None,
                            'predicted': self.request_value is None and self.predicted_value is not None},
                }
        if self.request_value is None and self.predicted_value is None:
            info['error'] = 'Parameter not given and cant be parsed from pdf'
        return info

    def get(self):
        report = self._report()
        if self.request_value is not None:
            return self.request_value, report
        elif self.predicted_value is not None:
            return self.predicted_value, report
        else:
            return None, report


class SizeParameter(ParameterI):
    def is_valid(self, size):
        try:
            sep = '-'
            assert size is not None
            assert isinstance(size, str)
            assert len(size.split(sep)) == 3
            assert all(map(is_num, size.split(sep)))
            return True
        except AssertionError:
            return False

    def _parse(self, size):
        sizes = [float(x) for x in size.split('-')]
        return sizes


class MassParameter(ParameterI):
    def is_valid(self, mass):
        try:
            assert mass is not None
            assert isinstance(mass, str)
            assert all(map(is_num, mass))
            assert float(mass.replace(',', '.')) > 0
            return True
        except AssertionError:
            return False

    def _parse(self, mass):
        return float(mass.replace(',', '.'))


class MaterialParameter(ParameterI):
    def is_valid(self, material):
        try:
            assert material is not None
            assert isinstance(material, str)
            assert len(material) > 0
            return True
        except AssertionError:
            return False

    def _parse(self, material):
        return str(material)


class ThicknessParameter(ParameterI):
    def is_valid(self, value):
        try:
            assert value is not None
            assert is_num(value)
            assert float(value) > 0
            return True
        except AssertionError:
            return False

    def _parse(self, value):
        return float(value)


class DetailNameParameter(ParameterI):
    def is_valid(self, value):
        try:
            assert value is not None
            assert isinstance(value, str)
            assert len(value) > 0
            return True
        except AssertionError:
            return False

    def _parse(self, value):
        return str(value)