from math import ceil
import copy
from typing import List, Union
from service.models.packing.detail import DetailPolygonal
from service.models.packing.svg_nest_packing.packing import SvgNestPacker
from service.models.packing.utils.errors import PackingError


class ComposedDetail:
    def __init__(self, w: int, h: int, kim: float, ids: dict):
        print(f'  Created detail size ({w}, {h}) of {sum(ids.values())} details ids {ids}')
        """
        Metadetail composed of simple details. Currently has no visual property
        @param w   width of compose detail container
        @param h   height of compose detail container
        @param kim useful utilization of w*h list
        @param ids map ids:count of details placed on list
        """
        self.w = w
        self.h = h
        self.sum_square = kim*w*h
        self.ids = ids
        self.kim = kim

    @classmethod
    def from_details_list(cls, w, h, details: List[Union[DetailPolygonal, ]], iterations=2, rotations=5):
        print(f'Try to create detail size ({w},{h}), of ({len(details)} types, {sum(list(map(lambda d: d.quantity, details)))} fact) details')
        """
        Selects appropriate details sublist (balance between quantity and diversity,
        composes it on list WxH. Then chooses composition variant with the highest
        kim and creates cls instance
        """
        details = list(map(copy.deepcopy, details))
        details = list(filter(lambda d: d.fits((w, h)), details))
        assert len(details) > 0
        # Create details subset
        details = list(filter(lambda d: d.get_square() <= w*h, details))
        optimal_qs = ComposedDetail._choose_quantities_strategy(w*h, details)
        for i, detail in enumerate(details):
            details[i].quantity = optimal_qs[detail.idx]


        # Compose with svgNest
        packer = SvgNestPacker()
        print(f'Start packing of ({len(details)} types, {sum(list(map(lambda d: d.quantity, details)))} fact) details')
        pack_params = dict(details=details,
                           material={'width': w, 'height': h},
                           iterations=iterations, rotations=rotations,
                           render=False)
        try:
            packing_info = packer(**pack_params)
        except PackingError:
            print('Internal svgNest packing error')
            raise PackingError

        # Select best compose detail
        best_kim = -0.1
        best_packmap_ix = -1
        for i in range(packing_info['results']['materials']['n']):
            if packing_info['results']['kim']['all'][i] > best_kim:
                best_kim = packing_info['results']['kim']['all'][i]
                best_packmap_ix = i

        # create instance
        ids = packing_info['results']['ids_per_list'][best_packmap_ix]
        return cls(w, h, best_kim, ids)

    @classmethod
    def from_detail_and_list(cls, detail, details: List[Union[DetailPolygonal, ]], iterations=2, rotations=5):
        print(f'Try to append to detail size ({detail.w},{detail.h}), with ({len(details)} types, {sum(list(map(lambda d: d.quantity, details)))} fact) details')
        """
        Selects appropriate details sublist (balance between quantity and diversity,
        composes it on list WxH width given (!) high-kim detail. Then chooses composition
        variant with the highest kim and creates cls instance
        """
        details = list(map(copy.deepcopy, details))
        detail = copy.deepcopy(detail)
        if len(details) == 0:
            return cls(detail.w, detail.h, detail.kim, {detail.idx: 1})

        # Create details subset
        wh = detail.w * detail.h
        free_area = (1 - detail.kim) * wh
        main_detail_idx = detail.idx

        details = list(filter(lambda d: d.get_square() <= free_area, details))
        optimal_qs = ComposedDetail._choose_quantities_strategy(free_area, details)
        for i, detail in enumerate(details):
            details[i].quantity = optimal_qs[detail.idx]

        if len(details) == 0:
            return cls(detail.w, detail.h, detail.kim, {detail.idx: 1})

        # Compose with svgNest
        packer = SvgNestPacker()
        # As far as main detail need to be packed with any sand detail, we can increase
        # chance to find good packing if packer would have copy of main detail for every
        # subset of sand details
        detail.quantity = len(details)
        details += [detail]
        pack_params = dict(details=details,
                           material={'width': detail.w, 'height': detail.h},
                           iterations=iterations, rotations=rotations, render=False)
        try:
            packing_info = packer(**pack_params)
        except PackingError:
            print('Internal svgNest packing error')
            raise PackingError

        # Select best compose detail
        best_kim = -0.1
        best_packmap_ix = -1
        # assert packing_info['results']['materials']['n'] >= 1
        for i in range(packing_info['results']['materials']['n']):
            if main_detail_idx not in packing_info['results']['ids_per_list'][i]:
                # dont need compose detail without main detail
                continue
            if packing_info['results']['kims']['all'][i] > best_kim:
                best_kim = packing_info['results']['kims']['all'][i]
                best_packmap_ix = i

        if best_kim > 0:
            # create instance
            ids = packing_info['results']['ids_per_list'][best_packmap_ix]
            print(f"{sum(ids.values())} details packed with target detail")
            return cls(detail.w, detail.h, best_kim, ids)
        else:
            # Compose detail will consist of only one detail
            print("No details packed with target detail((")
            return cls(detail.w, detail.h, detail.kim, {detail.idx: 1})


    @classmethod
    def _choose_quantities_strategy(cls, free_area, details: [List[DetailPolygonal]]):
        assert all(list(map(lambda d: d.get_square() <= free_area, details)))
        print('cast')
        optimals = {}
        for i, detail in enumerate(details):
            need_to_fullfill = ceil(free_area / detail.get_square())
            optimal_q = ceil(need_to_fullfill / len(details)) + 1
            print(f'  {detail.dxf.name}   optimal {optimal_q}  has {detail.quantity}')
            optimal_q = min(detail.quantity, optimal_q)
            optimals[detail.idx] = optimal_q
        return optimals