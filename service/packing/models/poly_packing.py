from math import ceil
from collections import Counter

from .utils.arq_utils import Numbers
from .utils.math import round_arr, mean, mul
from .utils.save import save_rendered_packings
from .polygon_packing_model.figure import Figure
from .polygon_packing_model.packmap import Packmap
from .polygon_packing_model.pack_strategies import pack_4_sides as pack_strategy

RETURN_GIVEN_DATA = False


class PackError(Exception):
    pass


def pack_polygonal(params):
    '''
    Funciton that parses parameters of packing and packs items with no-fit-polygon algorithm
    @param params:  PackingParameters object
    @returns:  json with errors or packing results information
    '''

    # parse args
    if len(params.errors) > 0:
        return {'errors': params.errors, 'warnings': params.warnings}

    # fetch parsed args
    visualize = params.visualize
    details = params.details
    material = params.material
    material_shape = (params.material_width, params.material_height)

    # calc additional metrics
    details_square = sum(list(map(lambda d: d.get_details_square(), details)))
    material_square = mul(material_shape)
    ideal_lists_n = details_square / material_square

    # validate details sizes
    if not params.all_details_fits_material():
        return {'errors': params.errors, 'warnings': params.warnings}

    # feed to model
    try:
        result = fit_nfp(details, material_shape, visualize=visualize)
        materials_n = result['lists']
        kims_per_list = result['results']
        visualizations = result['visualizations']
        inserted_idxs = result['inserted_on_lists']
        if materials_n == -1:
            raise PackError
    except PackError:
        error = {'pack_error': 'cant pack given details with such material'}
        return {'errors': params.errors + [error], 'warnings': params.warnings}

    # fetch packing data
    kims_per_list = round_arr(kims_per_list)
    avg_kim = mean(kims_per_list)
    archive_name = save_rendered_packings(visualizations)

    result = {'errors': [], 'warnings': params.warnings,
              'additional': {'details_square': Numbers.shortify(details_square),
                             'material_square': Numbers.shortify(material_square),
                             'ideal_lists_n': ceil(ideal_lists_n)},
              'results': {'materials': {'n': materials_n},
                          'kim': {'average': avg_kim,
                                  'all': kims_per_list},
                          'ids_per_list': inserted_idxs},
              'renders': archive_name}

    if RETURN_GIVEN_DATA:
        result['args'] = request.args
        result['json'] = request.json

    return result


def fit_nfp(details, material, visualize=False):
    '''
    Packs details on materials lists. Trying to minimize lists number
    @param details    list of Detail
    @param material   pair of material parameters (width, height)
    @param visualize  return images with packing or not
    @return           number of material lists used to pack given details
    '''
    # convert Details to Figures and flat the list
    figures = []
    for detail in details:
        figures.append(Figure(detail.load_dxf_points(), detail))

    lists = 0
    results = []
    visualizations = []
    inserted_on_lists = []
    get_total_q = lambda figs: sum(list(map(lambda fig: fig.detail.quantity, figs)))
    prev_total_q = -1

    # fit details with minimum lists of material
    while lists == 0 or get_total_q(figures):
        packmap, figures, inserted_idxs = pack_figures(figures, material)

        inserted_on_lists.append(inserted_idxs)
        if visualize:
            visualizations.append(packmap.render_full_packmap())

        kim = packmap.calc_packmap_kim() * 100
        results.append(kim)

        lists += 1
        if lists % 10 == 0:
            print(lists, 'lists used', get_total_q(figures), 'left to insert')

        if get_total_q(figures) == 0:
            print('no details left to pack')
            break

        if lists >= 10000 or get_total_q(figures)==prev_total_q:
            print('Endless error')
            return  {'lists': -1,
                     'results': [],
                     'visualizations': [],
                     'inserted_on_lists': []}
        prev_total_q = get_total_q(figures)

    if results[-1] == 0.:
        results = results[:-1]
        lists -= 1

    return {'lists': lists,
            'results': results,
            'visualizations': visualizations,
            'inserted_on_lists': inserted_on_lists}


def pack_figures(figures, material):
    '''
    Packs given figures on one material list until it can be performed.
    Figures are list of unique figures with quantity parameters that 
    specifies number of it's instances to pack
    '''
    figures = sorted(figures, key=lambda fig: (-fig.get_shape()[0], -fig.get_shape()[1]))
    inserted = Counter()
    packmap = Packmap(*material)
#     print('left to pack %d figures' % )

    for fig in figures:
        while fig.detail.quantity > 0:
            res_fig, res_kim = pack_strategy(packmap, fig.copy(), )
            if not res_fig:
                # stop pack current figure type and try the next shape
                print('cant insert fig #%d' % fig.detail.idx)
                break
            packmap.add_polygon(res_fig)
            print('pack res fig #%d'% fig.detail.idx, res_kim)
            fig.detail.decrease(1)
            inserted[int(fig.detail.idx)] += 1

    return packmap, figures, inserted

