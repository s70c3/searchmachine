from . import extract_table
import cv2
import numpy as np
from skimage.morphology import skeletonize


def _get_all_coords(lines, axis):
    g = lines
    if axis == 'x':
        return [l.x1 for l in g] + [l.x2 for l in g]
    else:
        return [l.y1 for l in g] + [l.y2 for l in g]


def _find_avg_coord(lines, axis):
    vs = _get_all_coords(lines, axis)
    return int(sum(vs) / len(vs))


def _find_min_coord(lines, axis):
    return min(_get_all_coords(lines, axis))


def _find_max_coord(lines, axis):
    return max(_get_all_coords(lines, axis))


def _is_btwn(line, coord, eps=0):
    return coord >= line.x1 - eps and coord <= line.x2 + eps


def _is_intersect(line1, line2, eps=0):
    return _is_btwn(line1, line2.x1, eps) or _is_btwn(line1, line2.x2, eps)


def _join_hlines(lines):
    g = lines
    y = _find_avg_coord(g, 'y')
    return extract_table.Line(min([l.x1 for l in g]), y, max([l.x2 for l in g]), y)


def _join_vlines(lines):
    x = _find_avg_coord(lines, 'x')
    return extract_table.Line(x, _find_min_coord(lines, 'y'), x, _find_max_coord(lines, 'y'))


def _combine_neighb_hlines(hlines, eps):
    parts = hlines[:]
    splits = []
    while len(parts) > 0:
        new_parts = []
        main = parts[0]
        split = [main]
        for part in parts[1:]:
            if _is_intersect(main, part, eps):
                split.append(part)
            else:
                new_parts.append(part)
        splits.append(_join_hlines(split))
        parts = new_parts
    return splits


def _split_hgroup(g, line_gap):
    hlines = g
    splits = _combine_neighb_hlines(hlines, line_gap)
    while not len(hlines) == len(splits):
        hlines = splits
        splits = _combine_neighb_hlines(hlines, line_gap)
    return hlines


def _group_vlines(vlines, consts):
    vgroups = []
    group = [vlines[0]]
    for l in vlines[1:]:
        if consts.equivalent(l.x1, group[0].x1):
            group.append(l)
        else:
            vgroups.append(group)
            group = [l]
    vgroups.append(group)
    return [_join_vlines(g) for g in vgroups]


def _group_hlines(hlines, consts, W):
    hgroups = []
    group = [hlines[0]]
    for l in hlines[1:]:
        if consts.equivalent(l.y1, group[0].y1):
            group.append(l)
        else:
            hgroups.append(group)
            group = [l]
    hgroups.append(group)
    # split hgroups
    splits = [_split_hgroup(g, W // 5) for g in hgroups]
    hgroups = []
    for s in splits: hgroups += s
    return hgroups


def _get_lines(img):
    minLineLength = 30
    maxLineGap = 5

    #  Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(img, 2, np.pi / 180, 50, None, minLineLength, maxLineGap)
    lines = [extract_table.create_line_from_HoughP(l) for l in lines]

    # select vlines & hlines
    hlines = sorted(filter(lambda x: x.is_horizontal(), lines), key=lambda x: x.mean_y())
    vlines = sorted(filter(lambda x: x.is_vertical(), lines), key=lambda x: x.mean_x())
    return hlines, vlines


def _elongate_hline(l, W):
    if abs(l.x1 - l.x2) > W // 2:
        return extract_table.Line(0, l.y1, W, l.y2)
    else:
        return l


def _add_lines(img, lines, color):
    for l in lines: img = cv2.line(img, (l.x1, l.y1), (l.x2, l.y2), color)
    return img


def get_clean_table_img(table_img, consts):
    H, W = table_img.shape

    skel = skeletonize(table_img / 255).astype(np.uint8) * 255
    hlines, vlines = _get_lines(skel)
    hlines = _group_hlines(hlines, consts, W)
    vlines = _group_vlines(vlines, consts)
    hlines = [_elongate_hline(l, W) for l in hlines]

    I = skel * 0
    I = _add_lines(I, hlines, 255)
    I = _add_lines(I, vlines, 255)
    return I