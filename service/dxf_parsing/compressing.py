import numpy as np

def points_sequence_to_lines(contour):
    """
    Reduces chains of 3+ sequential points with diff only in one
    coordinate down to 2 points
    @param contour: dxf contour
    @return: contour: compressed dxf contour
    """
    assert isinstance(contour, np.ndarray)
    assert len(contour.shape) == 2
    assert contour.shape[1] == 2

    def is_line(c1, c2):
        if abs(c1[0] - c2[0]) >= 1 and abs(c1[1] - c2[1]) == 0:
            return True
        if abs(c1[1] - c2[1]) >= 1 and abs(c1[0] - c2[0]) == 0:
            return True
        return False

    contour = contour.tolist()
    new_contour = []
    prev_coord = contour[0]
    for i, coord in enumerate(contour[1:], start=1):
        # if current point breaks straight line, call it a node
        # and append to polyline contour. Try to examine next line
        # from this breakpoint
        if not is_line(prev_coord, contour[i]):
            new_contour.append(contour[i])
            prev_coord = contour[i]

    new_contour = np.array(new_contour)
    return new_contour



def skew_lines_to_pair_points(contour):
    def get_k(p1, p2):
        is_vertical = lambda p1, p2: p1[0] == p2[0]
        is_horizontal = lambda p1, p2: p1[1] == p2[1]
        is_same = lambda p1, p2: p1 == p2

        if is_vertical(p1, p2) or is_same(p1, p2):
            return None
        if is_horizontal(p1, p2):
            return 0
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        k = dy / dx
        return k

    contour = contour.tolist()
    filtred_points = []

    line_origin = contour[0]
    for i, point in enumerate(contour[1:-1], start=1):
        k_origin = get_k(line_origin, point)
        k_skip = get_k(line_origin, contour[i+1])
        if k_origin != k_skip:
            filtred_points.append(point)
            line_origin = point
    filtred_points.append(contour[-1])
    return np.array(filtred_points)



def filter_points_duplicates(contour):
    """
    Deletes sequential duplicating points from contour
    @param contour: dxf contour
    @return: contour: compressed dxf contour
    """
    assert isinstance(contour, np.ndarray)
    assert len(contour.shape) == 2
    assert contour.shape[1] == 2

    def is_duplicate(c1, c2):
        return c1 == c2

    contour = contour.tolist()
    new_contour = []
    prev_coord = contour[0]
    for i, coord in enumerate(contour[1:], start=1):

        if not is_duplicate(prev_coord, contour[i]):
            new_contour.append(contour[i])
            prev_coord = contour[i]

    new_contour = np.array(new_contour)
    return new_contour