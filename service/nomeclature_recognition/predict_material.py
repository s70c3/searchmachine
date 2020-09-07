from nomeclature_recognition.bbox import Bbox, TextBbox

def _wb_to_tbbox(wb):
    (x0, y0), (x1, y1) = wb.position
    bbox = Bbox(x0=x0, y0=y0, x1=x1, y1=y1)
    text_bbox = TextBbox(bbox=bbox, text=wb.content)
    return text_bbox

def _sort_left_to_right(tbs):
    return sorted(tbs, key=lambda x: x.bbox.x0)

def _y0_stats(tbs):
    min_y0 = min([tb.bbox.y0 for tb in tbs])
    max_y0 = max([tb.bbox.y0 for tb in tbs])
    middle_y0 = (min_y0 + max_y0) / 2
    return min_y0, max_y0, middle_y0

def _select_middles(tbs):
    min_y0, max_y0, middle_y0 = _y0_stats(tbs)
    return [
        tb for tb in tbs
        if abs(tb.bbox.y0-middle_y0) < abs(tb.bbox.y0-min_y0)
            and abs(tb.bbox.y0-middle_y0) < abs(tb.bbox.y0-max_y0)]

def _split_in_2_by_y0(tbs, y0_1, y0_2):
    criterion = lambda x: abs(x.bbox.y0-y0_1) <= abs(x.bbox.y0-y0_2)
    tbs1 = list(filter(criterion, tbs))
    tbs2 = list(filter(lambda x: not criterion(x), tbs))
    return tbs1, tbs2

def _split_bboxes(tbs):
    min_y0, max_y0, middle_y0 = _y0_stats(tbs)

    middle_tbs = _select_middles(tbs)
    if len(middle_tbs) > 0:
        upper, _ = _split_in_2_by_y0(tbs, min_y0, middle_y0)
        lower, _ = _split_in_2_by_y0(tbs, max_y0, middle_y0)
        leftm = _sort_left_to_right(middle_tbs)[0]
        upper += [leftm]
    else:
        upper, lower = _split_in_2_by_y0(tbs, min_y0, max_y0)

    return upper, lower

def _join_text(tbs):
    tbs = _sort_left_to_right(tbs)
    return ' '.join(map(lambda x: x.text, tbs))

def combine_text(wboxes):
    tbboxes = list(map(_wb_to_tbbox, wboxes))
    upper, lower = _split_bboxes(tbboxes)
    utext = _join_text(upper)
    ltext = _join_text(lower)
    return utext, ltext