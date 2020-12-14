from math import log2, pi
from tqdm import tqdm_notebook as tqdm

from .figure import Figure


def main(packmap, figures, pack_f, logging=True):
    figures = sorted(figures, key=lambda f: -f.area)
    for i, p in tqdm(enumerate(figures), total=len(figures)):
        best_fig, best_res = pack_f(packmap, p)
        
        if best_fig is not None:
            packmap.add_polygon(best_fig)
            if logging:
                print('[%d/%d] insert with kim' % (i, len(figures)), best_res)
        else:
            if logging:
                print('[%d/%d] cant pack figure' % (i, len(figures)))


def pack_1_sides(m, p):
    best_fig = None
    best_res = -1
    maybe_to_list = lambda x: x if isinstance(x, list) else [x]
    points = p.get_points()

    stepsize = 1
    for j in range(0, len(p), stepsize):
        curr_p = Figure(maybe_to_list(points[j:]) + maybe_to_list(points[:j]))
        fig, res = m.get_best_kim_coord(curr_p)
        if res:
            if res > best_res:
                best_res = res
                best_fig = fig
    return best_fig, best_res


def pack_2_sides(m, p):
    best_fig = None
    best_res = -1
    maybe_to_list = lambda x: x if isinstance(x, list) else [x]
    points = p.get_points()
    
    for deg in [0, -180]:
        stepsize = 1 if len(p) <= 20 else int(len(p)//log2(len(p))*2)
        for j in range(0, len(p), stepsize):
            curr_p = Figure(maybe_to_list(points[j:]) + maybe_to_list(points[:j]))
            curr_p = curr_p.rotate(deg)
            fig, res = m.get_best_kim_coord(curr_p)
            if res:
                if res > best_res:
                    best_res = res
                    best_fig = fig
    return best_fig, best_res


def pack_4_sides(m, p):
    best_fig = None
    best_res = -1
    maybe_to_list = lambda x: x if isinstance(x, list) else [x]
    points = p.get_points()
    
    for deg in [0, -180, 90, -90]:
        stepsize = 1 if len(p) <= 20 else int(len(p)//log2(len(p))*3)
        for j in range(0, len(p), stepsize):
            curr_p = Figure(maybe_to_list(points[j:]) + maybe_to_list(points[:j]))
            curr_p = curr_p.rotate(deg)
#             curr_p = curr_p.move(1, 0)
            fig, res = m.get_best_kim_coord(curr_p)
            if res:
                if res > best_res:
                    best_res = res
                    best_fig = fig
    return best_fig, best_res
    
    
