import random
from pprint import pprint as print
import yaml
from .test_env import send_post_request
from .utils import check_field

config = yaml.load(open('./tests/tests_config.yaml').read(), Loader=yaml.FullLoader)
ADDRESS = config['address'] + config['rect_packing']['method_name']


def test_1():
    params = {'details': [{'height': 50, 'width': 142, 'quantity': 112},
              {'height': 1665, 'width': 60, 'quantity': 1},
              {'height': 1757, 'width': 349, 'quantity': 3}],
             'render_packing_maps': True,
             'material': {'height': 3000, 'width': 1200}}
    res = send_post_request(ADDRESS, params)
    assert check_field(res, 'errors')
    assert check_field(res, 'warnings')
    assert len(res['errors']) == 0, "This test should be passed correctly"
    assert check_field(res, 'renders')
    assert check_field(res, 'results')


def test_2():
    WIDTH_LIMITS = (10, 400)
    HEIGHT_LIMITS = (10, 400)
    QUANTITY_LIMITS = (1, 30)

    def gen_detail():
        w = random.randint(*WIDTH_LIMITS)
        h = random.randint(*HEIGHT_LIMITS)
        q = random.randint(*QUANTITY_LIMITS)
        return {'width': w, 'height': h, 'quantity': q}
    
    params = {'details': [gen_detail() for _ in range(50)],
             'render_packing_maps': True,
             'material': {'height': 3000, 'width': 1200}}
    
    res = send_post_request(ADDRESS, params)
    assert check_field(res, 'errors')
    assert check_field(res, 'warnings')
    assert len(res['errors']) == 0, "This test should be passed correctly"
    assert check_field(res, 'renders')
    assert check_field(res, 'results')
    
    
def main():
    test_1()
    test_2()