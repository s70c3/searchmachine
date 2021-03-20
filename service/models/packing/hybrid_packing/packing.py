from statistics import mean
from collections import Counter, namedtuple
from typing import List

from .kd_tree import KDTree
from service.models.packing.utils.errors import PackingError
from service.models.packing.detail import DetailPolygonal
from service.models.packing.stamper import Stamper, CachiedStamper
from .composed_detail import ComposedDetail

mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr)>1 else arr[0]
stamper = Stamper()
def det2ids(details: List[DetailPolygonal]):
    return {d.idx: d.quantity for d in details}


class HybridPacker:
    def __call__(self, details: List[DetailPolygonal], material, iterations, rotations):
        '''
        Funciton that parses parameters of packing and packs items with kd tree
        '''
        # fetch parsed args
        material_shape = (material['width'], material['height'])

        # feed to model
        try:
            cachied_stamper = CachiedStamper()
            info, visualizations = self._fit_trees(details, material_shape, cachied_stamper, visualize=False)
            info['kim']['all'] = list(map(lambda n: round(n, 2), info['kim']['all']))
        except PackingError:
            error = {'pack_error': 'cant pack given details with such material'}
            return {'errors': [error]}

        # archive_name = save_rendered_packings(visualizations, name_suffix='rect')
        return {'results': info,
                'renders': 'Not implemented'}

    def _fit_trees(self, details: List[DetailPolygonal], material, cachied_stamper, visualize=False):
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
            w, h = detail.get_shape()
            if h < w:
                detail.rotate_()
                w, h = h, w
            if w > material[0] or h > material[1]:
                detail.rotate_()

        prev_q = -1
        PrevPacking = namedtuple("PrevPacking", ['inserted', 'visual'])
        prev_pack = None
        # fit details with minimum trees
        while lists == 0 or get_total_q(details) > 0:
            tree = KDTree((0, 0), material)

            # delete 0-quantity details
            for i, detail in enumerate(details):
                assert detail.quantity >= 0
                if detail.quantity == 0:
                    details[i] = None
            details = list(filter(lambda e: e is not None, details))

            square_before = get_total_square(details)
            unused_details_ids = {d.idx:d.quantity for d in details}
            if (prev_pack is not None) and (stamper.can_stamp(prev_pack.inserted, unused_details_ids)):
                # decrease details same as in pattern list
                details = self._stamp_list(details, prev_pack.inserted)
                inserted, visual = prev_pack
                print('Stamp list')
            else:
                details, visual, inserted = self._max_fill_tree_fast(tree, details, cachied_stamper, visualize=visualize)
            prev_pack = PrevPacking(inserted, visual)

            # if visual is not None:
            #     visualizations.append(visual)
            inserted_on_lists.append(inserted)

            square_inserted = square_before - get_total_square(details)
            perc = square_inserted / (material[0] * material[1]) * 100
            results.append(perc)

            lists += 1
            if lists % 10 == 0:
                print(lists, 'lists used', get_total_q(details), 'left to insert')

            if get_total_q(details) == 0:
                print('cant insert any detail')
                break

            if lists >= 10000 or prev_q == get_total_q(details):
                print('Endless error')
                raise PackingError
            prev_q = get_total_q(details)

        if results[-1] == 0.:
            results = results[:-1]
            lists -= 1

        info = {'materials': {'n': lists},
                'kim': {'average': mean(results),
                        'all': results},
                'ids_per_list': inserted_on_lists}
        return info, visualizations

    def _stamp_list(self, details: List[DetailPolygonal], pattern_ids: dict):
        for idx, n in pattern_ids.items():
            for j, det in enumerate(details):
                if det.idx == int(idx):
                    try:
                        details[j].decrease_(n)
                    except Exception as e:
                        print('>> Cant descrease', det.dxf.name, 'q', det.quantity, 'for n', n)
                        raise e
                    break
        return details

    def _max_fill_tree_fast(self, tree, details: List[DetailPolygonal], cachied_stamper, printing=False, visualize=False):
        """Fills one tree with given elements until in possible"""
        details = sorted(details, key=lambda e: (-e.h, -e.w))
        visual = None
        inserted = Counter()

        for i, detail in enumerate(details):
            while detail.quantity > 0:
                N_START = sum(list(map(lambda d: d.quantity, details)))

                if detail.get_kim() > 0.9:
                    other_details = details[:i] + details[i+1:]
                    other_details = list(filter(lambda d: d.quantity>0, other_details))

                    cachied_detail = cachied_stamper.search_cachied_compose(detail.w, detail.h, det2ids(other_details), required_detail=detail)
                    if cachied_detail is not None:
                        composed = cachied_detail
                        print('Take detail from cache')
                    else:
                        composed = ComposedDetail.from_detail_and_list(detail, other_details)
                        if len(composed.ids) > 1:
                            cachied_stamper.add(composed)
                else:
                    w, h = tree.get_biggest_free_rect()
                    if w*h > 1000*500: # speedup heuristic
                        w = w//2
                        h = h//2

                    other_details = list(filter(lambda d: d.quantity > 0, details))
                    other_details = list(filter(lambda d: d.fits((w, h)), other_details))
                    if len(other_details) == 0:
                        break

                    cachied_detail = cachied_stamper.search_cachied_compose(w, h, det2ids(other_details))
                    if cachied_detail is not None:
                        composed = cachied_detail
                        print('Take detail from cache')
                    else:
                        composed = ComposedDetail.from_details_list(w, h, other_details)
                        cachied_stamper.add(composed)
                res = tree.insert(composed.w, composed.h, composed)
                if not res:
                    break # stop fill tree with shape and try next shape

                print('Insert composed with kim', composed.kim, 'ids', composed.ids)
                # if composed detail inserted, update queue for insertion
                for idx, n in composed.ids.items():
                    for j, det in enumerate(details):
                        if det.idx == int(idx):
                            try:
                                details[j].decrease_(n)
                            except Exception as e:
                                print('>> Cant descrease', det.dxf.name,'q', det.quantity,'for n',n)
                                raise e
                            inserted[int(detail.idx)] += n
                            break

                N_STOP = sum(list(map(lambda d: d.quantity, details)))
                assert N_STOP < N_START

        return details, visual, inserted
