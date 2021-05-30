import cv2
import numpy as np
from .detection import *
from .show import show


def hough_cleaning(img):
    img = img.copy()
    edges = cv2.Canny(img, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=30)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return img


def process_morph(img, kernel):
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    k = np.ones((kernel, kernel))
    img = cv2.dilate(img, kernel=k)
    img = cv2.erode(img, kernel=k)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return img


def get_thick_contour(img, threshold=0.15, kernel_stick=55, kernel=7, fixKernel=False):
    # преобразование для удаления тонких деталей с заданным ядром kernel.
    # обычно успех достигается с ядром 3-4
    img_re = process_morph(img, kernel)

    # все существующие контуры утолщаем как базу
    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_re, contours[1:], -1, 0, 5)

    # преобразование для склеивания. ядро берется из параметров - по умолчанию 55,
    # подобранное перебором оптимальное значение
    k = np.ones((kernel_stick, kernel_stick))

    img_re = cv2.erode(img_re, kernel=k)
    img_re = cv2.dilate(img_re, kernel=k)

    # белая рамка для того, чтобы отсчечь контуры, прилипшие к краю.
    cv2.rectangle(img_re, (0, 0), (img_re.shape[1], img_re.shape[0]), (255, 255, 255), 60)
    # ищем замкнутые контуры
    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    # если мы не нашли контуров, то идем на новую итерацию
    if len(contours) < 2 and kernel > 1:
        return get_thick_contour(img, kernel_stick=kernel_stick, threshold=threshold, kernel=kernel - 1)
    elif len(contours) < 2 and kernel == 1:
        return [], kernel

    # ищем максимальный по площади окружающего квадрата контур
    get_area = lambda c: -c[2] * c[3]
    contour_sizes = [(get_area(cv2.boundingRect(contour)), contour) for contour in contours]
    b_contour = sorted(contour_sizes, key=lambda x: x[0])[1]


    # выходим если мы достигли исходного изображения,
    # или если найденный контур превосходит порог
    # или если мы зафиксировали размер ядра (для уточнения)
    if kernel == 1 or (-b_contour[0] / img.size > threshold) or fixKernel:
        return b_contour[1], kernel
    else:
        return get_thick_contour(img, kernel_stick=kernel_stick, threshold=threshold, kernel=kernel - 1)


def get_detail_image(img, threshold=0.2, clean_small_size=0, kernel_stick=55):
    img_copy = img.copy()
    # очищение мелких контуров, таких как буквы, размеры, символы.
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) < clean_small_size:
            cv2.drawContours(img_copy, [cnt], -1, 255, -1)

    # Получение контура и маски изображения
    contour, kernel = get_thick_contour(img_copy, threshold=threshold, kernel_stick=kernel_stick)
    mask = np.ones(img_copy.shape, dtype=np.uint8) * 255
    cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)
    inner = cv2.bitwise_or(img, mask)
    # уточнение контура с помощью эрозии с меньшим ядром. мы перебираем значения,
    # пока не получим контур, близкий по площади к грубому. благодаря этому многий мусор отсекается.
    kernel_loop = 11
    precision_contour, _ = get_thick_contour(inner.copy(), kernel_stick=kernel_loop, threshold=threshold)

    while cv2.contourArea(precision_contour) / cv2.contourArea(contour) < 0.7 and kernel_loop < 55:
        precision_contour, _ = get_thick_contour(img_copy, kernel_stick=kernel_loop,
                                                 threshold=threshold, kernel=kernel, fixKernel=True)
        kernel_loop += 2

    # получаем новую маску детали
    mask = np.ones(img_copy.shape, dtype=np.uint8) * 255
    cv2.drawContours(mask, [precision_contour], -1, (0, 0, 0), -1)
    inner = cv2.bitwise_or(img, mask)

    inner = cv2.erode(inner, kernel=np.ones((3, 3)))

    # получаем максимальный внешний контур
    cv2.rectangle(inner, (0, 0), (inner.shape[1], inner.shape[0]), (255, 255, 255), 5)
    inner = cv2.threshold(inner, thresh=254, maxval=255, type=cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(inner, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if len(contour_sizes)>1:
        b_contour_area, contour = sorted(contour_sizes, key=lambda x: x[0], reverse=True)[1]
    else:
        b_contour_area, contour = sorted(contour_sizes, key=lambda x: x[0], reverse=True)[0]

    if b_contour_area / cv2.contourArea(precision_contour) < 0.8:
        contour = precision_contour
    # возвращаем обрезанное по контуру изображение и сам контур
    return inner, contour


# get circles
def hough_circle(img):
    img = hough_cleaning(img)
    rows = img.shape[0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=2, maxRadius=400)
    return circles


def draw_circles(img, circles, color, thickness):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, color, thickness)


def get_inner_thick(img, kernel=6):
    # preprocess images
    img_re = process_morph(img, kernel)
    img = hough_cleaning(img)
    # finding and make thicker closed contours
    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_re, contours[1:], -1, 0, 5)

    # make white border
    cv2.rectangle(img_re, (0, 0), (img_re.shape[1], img_re.shape[0]), (255, 255, 255), 5)

    # find closed contours
    contours, hierarchy = cv2.findContours(img_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > 1000, contours))
    # get out if we found nothing
    if len(contours) < 2 and kernel > 1:
        return get_inner_thick(img, kernel=kernel - 1)

    else:
        return contours[1:]


def get_detail_inner(img, contour, isCircle=False):
    img = img.copy()
    cv2.drawContours(img, contour, -1, 255, 20)

    if isCircle:
        contours = hough_circle(img.copy())
    else:
        contours = get_inner_thick(img)
    return contours


def draw_inner(img, cntr, isCircle, color=(128, 128, 128), thickness=5):
    if isCircle:
        draw_circles(img, cntr, color, thickness)
    else:
        cv2.drawContours(img, cntr, -1, color, thickness)


def get_arrows(img):
    img = img.copy()
    edges = cv2.Canny(img, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=30)
    return lines


def draw_lines(img, lines, color):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, 10)


def get_contours(img, outer, inner, arrows):
    detail, outer_contour = get_detail_image(img)
    if inner:
        # get holes. flag isCircle for better detecting circle-shape holes
        isCircle = True
        inner_contour = get_detail_inner(detail, outer_contour, isCircle)
    else:
        inner_contour = None
    # get holes via remove all predicted detail contours
    if arrows:
        arrows_img = img.copy()
        cv2.drawContours(arrows_img, outer_contour, -1, 255, 20)
        cv2.drawContours(arrows_img, [outer_contour], -1, 255, -1)
        draw_inner(arrows_img, inner_contour, isCircle, 255)
        lines = get_arrows(arrows_img)
    else:
        lines = None
    if not outer:
        outer_contour = None
    return outer_contour, inner_contour, lines


def draw_detail_contours(img, outer=None, inner=None, isCircle=True, arrows=None):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    if outer is not None:
        cv2.drawContours(img, outer, -1, (0, 0, 255), 5)
    if inner is not None:
        draw_inner(img, inner, isCircle, color=(255, 0, 0))
    if arrows is not None:
        draw_lines(img, arrows, (0, 255, 0))
    return img


def get_draw_countours(img, outer=False, inner=False, arrows=False):
    outer_contour, inner_contour, arrows = get_contours(img, outer, inner, arrows)
    return draw_detail_contours(img, outer_contour, inner_contour, True, arrows)


def get_contours_info(img, visualize=False):
    projections = crop_conturs(img)
    try:
        if visualize and projections is not None:
            for proj in projections:
                show(proj, get_draw_countours(proj, outer=True, inner=True))
        if projections is not None:
            projection_number = len(projections)
        else:
            projection_number = 0
        if projection_number>0:
            contours_lens = [cv2.arcLength(get_thick_contour(proj)[0],True) for proj in projections]
        else:
            contours_lens = np.NaN
        if projection_number>0 and hough_circle(projections[0]) is not None:
            holes_number = len(hough_circle(projections[0]))
        else:
            holes_number = 0
        return [projection_number, contours_lens, holes_number]
    except Exception as e:
        print ("countours", e)
        return [np.NaN, np.NaN, np.NaN]