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