import random
from pprint import pprint as print
import os
import yaml
from .test_env import send_post_request
from .utils import check_field

config = yaml.load(open('./tests/tests_config.yaml').read(), Loader=yaml.FullLoader)
ADDRESS = config['address'] + config['poly_packing']['method_name']


def test_1():
    dxfs = os.listdir('/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ')
    get_dxf = lambda: random.choice(dxfs)
    params = {'details': [{'dxf': get_dxf(), 'quantity': 1},
                          {'dxf': get_dxf(), 'quantity': 1},
                          {'dxf': get_dxf(), 'quantity': 1},
                          {'dxf': get_dxf(), 'quantity': 1} ],
                         'render_packing_maps': True,
                         'material': {'height': 3000, 'width': 2000}}
    res = send_post_request(ADDRESS, params)
    print(res)
    assert check_field(res, 'errors')
    assert check_field(res, 'warnings')
    assert len(res['errors']) == 0, "This test should be passed correctly"
    assert check_field(res, 'renders')
    assert check_field(res, 'results')


def test_2():
    dxfs = os.listdir('/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ')
    get_dxf = lambda: random.choice(dxfs)
    params = {'details': [{'dxf': '4498.25.15.312.dxf', 'quantity': 1}],
                         'render_packing_maps': True,
                         'material': {'height': 500, 'width': 500}}
    res = send_post_request(ADDRESS, params)
    print(res)
    assert check_field(res, 'errors')
    assert check_field(res, 'warnings')
    assert len(res['errors']) == 0, "This test should be passed correctly"
    assert check_field(res, 'renders')
    assert check_field(res, 'results')

def test_3():
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
    try:
        test_3()
        raise Exception('should failed')
    except AssertionError:
        pass