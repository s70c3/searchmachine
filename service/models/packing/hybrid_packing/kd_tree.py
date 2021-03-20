class KDTree:
    def __init__(self, xy_l, xy_r, data=None):
        # upper_left and bottom_right
        self.xy_l = xy_l
        self.xy_r = xy_r
        self.data = data

        self.is_list = True
        self.has_data = False

        self.child_l = None
        self.child_r = None

    def _is_root(self):
        return self.child_l == None and self.child_r == None

    def _get_square(self):
        w = self.xy_r[0] - self.xy_l[0]
        h = self.xy_r[1] - self.xy_l[1]
        return w * h

    def _get_rw(self):
        w = self.xy_r[0] - self.xy_l[0]
        h = self.xy_r[1] - self.xy_l[1]
        return w, h

    def _get_rect(self):
        return self.xy_l, self.xy_r

    def insert(self, r_w, r_h, data):
        assert r_w is not None
        assert r_h is not None

        # insert left?
        if self.child_l:
            if self.child_l.insert(r_w, r_h, data):
                return True

        # insert right?
        if self.child_r:
            if self.child_r.insert(r_w, r_h, data):
                return True

        # maybe we are in unsplitted cell
        if self._fits(r_w, r_h):
            if self.is_list and not self.has_data:
                self._insert(r_w, r_h, data)
                return True

        return False

    def _fits(self, r_w, r_h):
        w = self.xy_r[0] - self.xy_l[0]
        h = self.xy_r[1] - self.xy_l[1]
        return w >= r_w and h >= r_h

    def _insert(self, r_w, r_h, data) -> bool:
        if self.has_data:
            return False

        else:
            if self._fits(r_w, r_h):
                # split current node
                criterion = self.xy_l[1] + r_h

                child1_l = self.xy_l
                child1_r = (self.xy_r[0], criterion)
                child2_l = (self.xy_l[0], criterion)
                child2_r = self.xy_r

                self.child_l = KDTree(child1_l, child1_r)
                self.child_r = KDTree(child2_l, child2_r)
                self.is_list = False

                # split lchild and fill it's lchild
                criterion = self.child_l.xy_l[0] + r_w

                child1_l = self.child_l.xy_l
                child1_r = (criterion, self.child_l.xy_r[1])
                child2_l = (criterion, self.child_l.xy_l[1])
                child2_r = self.child_l.xy_r

                self.child_l.child_l = KDTree(child1_l, child1_r, data)
                self.child_l.child_r = KDTree(child2_l, child2_r)

                self.child_l.is_list = False
                self.child_l.child_l.has_data = True

                return True

            return False

    def get_biggest_free_rect(self):
        whs = []
        if self.child_l is not None:
            w, h = self.child_l.get_biggest_free_rect()
            whs.append((w, h))
        if self.child_r is not None:
            w, h = self.child_r.get_biggest_free_rect()
            whs.append((w, h))

        if self.child_l is None and self.child_r is None:
            w, h = self._get_rw()
            whs.append((w, h))

        whs = sorted(whs, key=lambda s: -s[0]*s[1])
        return whs[0]


    def __repr__(self, margin=0):
        tabs = ' ' * margin
        childl = self.child_l.__repr__(margin + 1) if self.child_l else '-'
        childr = self.child_r.__repr__(margin + 1) if self.child_r else '-'

        if childl != '-' and self.child_l.has_data:
            childl += ' ' + str(self.child_l._get_rw())
        if childr != '-' and self.child_r.has_data:
            childr += ' ' + str(self.child_r._get_rw())

        return 'Rect {size1}, {size2}. Children: \n{tabs}{c1}\n{tabs}{c2}'.format(tabs=tabs,
                                                                                  size1=str(self.xy_l),
                                                                                  size2=str(self.xy_r),
                                                                                  c1=childl,
                                                                                  c2=childr)


