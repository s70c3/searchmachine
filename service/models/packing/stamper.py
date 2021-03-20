class Stamper:
    def can_stamp(self, list_details: dict, unused_details: dict):
        # Checks if current list can be duplicted. Duplication needs
        # unused details to contain items with same (or more) quantity
        # as in the current list
        for idx, q in list_details.items():
            if not idx in unused_details or unused_details[idx] < q:
                return False
        return True


class CachiedStamper(Stamper):
    def __init__(self):
        self.cache = set()

    def add(self, composed_detail):
        self.cache.add(composed_detail)

    def _search_with_required_detail(self, detail, unused_details):
        for cachied in self.cache:
            if cachied.w == detail.w and cachied.h == detail.h:
                if detail.idx in cachied.ids:
                    needed_idxs = dict(cachied.ids)
                    del needed_idxs[detail.idx]
                    if self.can_stamp(needed_idxs, unused_details):
                        return cachied
        return None


    def search_cachied_compose(self, w, h, unused_details: dict, required_detail=None):
        if required_detail is not None:
            return self._search_with_required_detail(required_detail, unused_details)

        for cachied in self.cache:
            if cachied.w == w and cachied.h == h:
                if self.can_stamp(cachied.ids, unused_details):
                    return cachied
        return None