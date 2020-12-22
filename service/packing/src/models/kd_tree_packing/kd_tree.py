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
        return w*h
    
    def _get_rw(self):
        w = self.xy_r[0] - self.xy_l[0]
        h = self.xy_r[1] - self.xy_l[1]
        return w, h
    
    def _get_rect(self):
        return self.xy_l, self.xy_r
        
    def insert(self, r_w, r_h, data=None):
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

    def _insert(self, r_w, r_h, data=None):
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

    def __repr__(self, margin=0):
        tabs = ' ' * margin
        childl = self.child_l.__repr__(margin+1) if self.child_l else '-'
        childr = self.child_r.__repr__(margin+1) if self.child_r else '-'

        if childl != '-' and self.child_l.has_data:
            childl += ' ' + str(self.child_l._get_rw())
        if childr != '-' and self.child_r.has_data:
            childr += ' ' + str(self.child_r._get_rw())

        return 'Rect {size1}, {size2}. Children: \n{tabs}{c1}\n{tabs}{c2}'.format(tabs=tabs,
                                                                                    size1=str(self.xy_l),
                                                                                    size2=str(self.xy_r),
                                                                                    c1=childl,
                                                                                    c2=childr)

    

    

from collections import Counter
sum_square_lst = lambda wh_arr: sum(list(map(lambda wh: wh[0]*wh[1], wh_arr)))
# def sum_square_dict(wh_dict):
#     sq = 0
#     for shape, q in wh_dict.items():
#         sq += shape[0] * shape[1] * q
#     return sq
    

PADDING = 2
def max_fill_tree_fast(tree, elements, printing=False, visualize=False):
    # Fills one tree with given element until in possible
    # @param  tree       empty kd tree
    # @param  elements   dict of shapes to quantity
    # @param  dxfs       dict of shapes to dxf files names
    elements = sorted(elements, key=lambda e: (-e.h, -e.w))
    visual = None
    inserted = Counter()
    
    for elem in elements:
        while elem.quantity > 0:
            res = tree.insert(elem.w+PADDING, elem.h+PADDING, elem.dxf_name)
            if not res:
#                 print('cant insert')
                break # stop fill tree with shape and try next shape
            elem.decrease(1)
            inserted[int(elem.idx)] += 1
            
            
           
    if visualize:
        visual = visualize_tree(tree)
        
    return elements, visual, inserted
    


from PIL import Image, ImageDraw
from .convert_dxf import dxf2image
from collections import defaultdict
# DXF_PATH = '/data/detail_price/dxf_хпц/dxf_ХПЦ_ТВЗ/'
DXF_PATH = '/home/iria/CT_Lab/detail_kim/reports/compare_with_CAM/DXF/'
TMP = './tmp/tmp.png'

# scale coefficient for rendered maps. Improves performance but looses picture quality 
SCALE_IMGS_K = 0.5

def visualize_tree(tree):
    resize = lambda size, k: tuple(map(lambda s: int(s*k), size))
    img = Image.new('L', 
                    resize(tree.xy_r, SCALE_IMGS_K), 
                    color=255)
    dxfs_cache = defaultdict(lambda: None)
    
    def draw_rect(xy, xy1):
        # draw rectangle outilne for given shape
        draw = ImageDraw.Draw(img)
        x, y = xy
        x1, y1 = xy1
        params = list(map(int, (x, y, x1, y1)))
        draw.rectangle(params, fill=None, outline=0, width=2)
        
    def draw_dxf(xy, xy1, img_orig, dxf_img):
        # draw given dxf image inside element with given shape
        def need_rotate(img_wh, cell_wh):
            i = img_wh
            c = cell_wh
            return (i[1] is max(i) and c[1] is not max(c)) or \
                   (i[0] is max(i) and c[0] is not max(c))
        
        draw = ImageDraw.Draw(img_orig)
        x, y = xy
        x1, y1 = list(map(int, xy1))
        w, h = x1 - x, y1-y
        
        if need_rotate(dxf_img.size, img.size):
            dxf_img = dxf_img.rotate(90, expand=1)
            dxf_img = dxf_img.resize((h, w))
        else:
            dxf_img = dxf_img.resize((w, h))
        img_orig = img_orig.paste(dxf_img, (x, y))
        return img_orig

    def get_node_rect(node):
        coord1 = resize(node.xy_l, SCALE_IMGS_K)
        coord2 = resize(node.xy_r, SCALE_IMGS_K)
        data = node.data
        return coord1, coord2, data

    def visit(node, img):
        if node.child_l is not None:
            visit(node.child_l, img)

        if node.child_r is not None:
            visit(node.child_r, img)

        if node.has_data:
            coord, wh, dxf = get_node_rect(node)
            if dxf is not None:
                # fetch from cache or add to cache
                dxf_img = dxfs_cache[DXF_PATH+dxf]
                if dxf_img is None:
                    dxf_img = dxf2image(DXF_PATH+dxf, scale_koef=SCALE_IMGS_K)
                    dxfs_cache[DXF_PATH + dxf] = dxf_img
                
                img = draw_dxf(coord, wh, img, dxf_img)
                draw_rect(coord, wh)
            else:
                draw_rect(coord, wh)
        
    visit(tree, img)
    return img