from . import extract_table, clear_table
import cv2
from .bbox import get_bbox


class CellExtractor:
    def __init__(self, table, consts):
        self.table = table
        self.consts = consts

    def _almost_0(self, v):
        return self.consts.almost_0(v)

    def _equiv(self, v1, v2):
        return self.consts.equivalent(v1, v2)

    def _get_cells(self):
        contours, hierarchy = cv2.findContours(self.table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = list(map(get_bbox, contours))
        # filter out too large and lines
        H, W = self.table.shape
        table_area = H * W
        is_too_large = lambda x: x.area() > table_area / 2
        is_a_line = lambda x: self._almost_0(x.width()) or self._almost_0(x.height())
        is_a_cell = lambda x: not is_too_large(x) and not is_a_line(x)
        bboxes = list(filter(is_a_cell, bboxes))
        return bboxes

    def _get_left_neighbour(self, cell, bboxes):
        x0, y1 = cell.x0, cell.y1
        criterion = lambda x: self._equiv(x.x1, x0) and self._equiv(x.y1, y1)
        candidates = list(filter(criterion, bboxes))
        if len(candidates) == 1:
            return candidates[0]
        else:
            return None

    def get_material_name_detail_cells(self):
        bboxes = self._get_cells()

        # get lowest cells
        max_low = max([bbox.y1 for bbox in bboxes])
        lowest_cells = list(filter(lambda x: self._equiv(x.y1, max_low), bboxes))

        lowest_cells_left_to_right = sorted(lowest_cells, key=lambda x: x.x1)
        material_bbox = lowest_cells_left_to_right[-2]

        aligned = list(filter(lambda b: abs(b.x0 - material_bbox.x0) < self.consts.SAME_LINES_DIFF, bboxes))
        aligned = sorted(aligned, key=lambda b: b.y0)
        name_bbox, detail_bbox, material_bbox = aligned

        return name_bbox, detail_bbox, material_bbox

    def _get_rightest_cell(self, nrow, bboxes):
        # counting from bottom, starting at 0
        max_right = max([bbox.x1 for bbox in bboxes])
        rightest_cells = list(filter(lambda x: self._equiv(x.x1, max_right), bboxes))
        rightest_cells_top_to_bottom = sorted(rightest_cells, key=lambda x: x.y1)
        return rightest_cells_top_to_bottom[-(nrow + 1)]

    def get_mass_cell(self):
        bboxes = self._get_cells()
        second_row_rightest_cell = self._get_rightest_cell(2, bboxes)
        return self._get_left_neighbour(second_row_rightest_cell, bboxes)

    def get_mass_header_cell(self):
        bboxes = self._get_cells()
        third_row_rightest_cell = self._get_rightest_cell(3, bboxes)
        return self._get_left_neighbour(third_row_rightest_cell, bboxes)

def _get_bboxes(cell_extractor):
    mass_bbox = cell_extractor.get_mass_cell()
    mass_header_bbox = cell_extractor.get_mass_header_cell()
    name_bbox, detail_bbox, material_bbox = cell_extractor.get_material_name_detail_cells()
    return [mass_bbox, mass_header_bbox, name_bbox, detail_bbox, material_bbox]

def get_cell_imgs(img):
    t_vh = extract_table.remove_text(img)
    consts = extract_table.ImageConstants(img)
    
    table_bbox = extract_table.get_table(t_vh, consts)
    table_img = extract_table.get_subimg(t_vh, table_bbox)

    try:
        bboxes = _get_bboxes(CellExtractor(table_img, consts))
    except:
        table_img = clear_table.get_clean_table_img(table_img, consts)
        bboxes = _get_bboxes(CellExtractor(table_img, consts))

    combined_bboxes = [extract_table.combine_bboxes(table_bbox, b) for b in bboxes]
    subimgs = [extract_table.get_subimg(img, b) for b in combined_bboxes]
    mass_subimg, mass_header_subimg, name_subimg, detail_subimg, material_subimg = subimgs
    return mass_subimg, mass_header_subimg, name_subimg, detail_subimg, material_subimg