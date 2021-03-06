from math import ceil
from typing import List

from ..detail import DetailRect
from .kd_tree import KDTree, max_fill_tree_fast
from ..utils.math import mul, round_arr, mean
from ..utils.arq_utils import Numbers
from ..utils.save import save_rendered_packings
from service.models.packing.utils.errors import PackingError


def pack_rectangular(details: List[DetailRect], material, visualize):
    '''
    Funciton that parses parameters of packing and packs items with kd tree
    '''
    # fetch parsed args
    material_shape = (material['width'], material['height'])

    # calc additional metrics
    details_square = sum(list(map(lambda d: d.get_square_x_quantity(), details)))
    material_square = mul(material_shape)
    ideal_lists_n = details_square / material_square     

    # feed to model
    try:
        materials_n, kims_per_list, visualizations, inserted_idxs = fit_trees(details, material_shape, visualize=visualize)
    except PackingError:
        error = {'pack_error': 'cant pack given details with such material'}
        return {'errors': [error]}
        
    # fetch packing data
    kims_per_list = round_arr(kims_per_list)
    avg_kim = mean(kims_per_list)
    archive_name = save_rendered_packings(visualizations, name_suffix='rect')
                                               
    return   {'errors': [],
              'additional': {'details_square': Numbers.shortify(details_square),
                             'material_square': Numbers.shortify(material_square),
                             'ideal_lists_n': ceil(ideal_lists_n)},
              'results': {'materials': {'n': materials_n},
                          'kim': {'average': avg_kim,
                                  'all': kims_per_list},
                          'ids_per_list': inserted_idxs},
              'renders': archive_name}


def fit_trees(details: List[DetailRect], material, visualize=False):
    # TODO: refactor and make it a clear function
    '''
    packs details on materials lists. Trying to minimize lists number
    @param details    list of Detail
    @param material   pair of material parameters (width, height)
    @param visualize  return images with packing or not
    @return           number of material lists used to pack given details
    '''
    lists = 0
    results = []
    visualizations = []
    inserted_on_lists = []
    inserted = dict()
    get_total_q = lambda ds: sum(list(map(lambda d: d.quantity, ds)))
    get_total_square = lambda ds: sum(list(map(lambda d: d.get_square_x_quantity(), ds)))

    # rotate material
    material = (min(material), max(material))
    
    # rotate details for kd tree
    for detail in details:
        if detail.is_rect():
            continue
            
        w, h = detail.get_shape()
        if h < w:
            detail.rotate_()
            w, h = h, w
        if w > material[0] or h > material[1]:
            detail.rotate_()
            
    prev_q = -1
    # fit details with minimum trees
    while lists == 0 or get_total_q(details) > 0:
        tree = KDTree((0,0), material)
        
        # delete 0-quantity details
        for i, detail in enumerate(details):
            assert detail.quantity >= 0
            if detail.quantity == 0:
                details[i] = None
        details = list(filter(lambda e: e is not None, details))

        
        sqare_before = get_total_square(details)
        details, visual, inserted = max_fill_tree_fast(tree, details, visualize=visualize)
        if visual is not None:
            visualizations.append(visual)
        inserted_on_lists.append(inserted)
        
        square_inserted = sqare_before - get_total_square(details)
        perc = square_inserted / (material[0] * material[1]) * 100
        results.append(perc)

        lists += 1
        if lists % 10 == 0:
            print(lists, 'lists used', get_total_q(details), 'left to insert')

            
        if get_total_q(details) == 0:
            print('cant insert any detail')
            break#raise PackError
            
        if lists >= 10000 or prev_q == get_total_q(details):
            print('Endless error')
            raise PackingError
            # return -1, [], [], []
        prev_q = get_total_q(details)
    
    if results[-1] == 0.:
        results = results[:-1]
        lists -= 1
    
    return lists, results, visualizations, inserted_on_lists