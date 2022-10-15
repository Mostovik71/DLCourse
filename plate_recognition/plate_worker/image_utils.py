import numpy as np
import math
import cv2
from typing import List, Union
from scipy.spatial import ConvexHull


def distance(p0: List or np.ndarray, p1: List or np.ndarray) -> float:
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def linear_line_matrix(p0: List, p1: List, verbode: bool = False) -> np.ndarray:
    """
    Вычесление коефициентов матрицы, описывающей линию по двум точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    matrix_a = y1 - y2
    matrix_b = x2 - x1
    matrix_c = x2*y1-x1*y2
    if verbode:
        print("Уравнение прямой, проходящей через эти точки:")
        print("%.4f*x + %.4fy = %.4f" % (matrix_a, matrix_b, matrix_c))
        print(matrix_a, matrix_b, matrix_c)
    return np.array([matrix_a, matrix_b, matrix_c])





def find_distances(points: np.ndarray or List) -> List:
    """
    TODO: describe function
    """
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if i < cnt - 1:
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1]), "p0": p0, "p1": p1,
                          "matrix": linear_line_matrix(points[p0], points[p1]),
                          "coef": fline(points[p0], points[p1])})
    return distanses


def fline(p0: List, p1: List, debug: bool = False) -> List:
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if x1 - x2 == 0:
        k = math.inf
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k*x2
    if debug:
        print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if a < 0:
        a180 = 180 + a
    return [k, b, a, a180, r]


def build_perspective(img: np.ndarray, rect: list, w: int, h: int) -> List:
    """
    image perspective transformation
    """
    img_h, img_w, img_c = img.shape
    if img_h < h:
        h = img_h
    if img_w < w:
        w = img_w
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    moment = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, moment, (w, h))


def get_cv_zone_rgb(img: np.ndarray, rect: list, gw: float = 0, gh: float = 0,
                    coef: float = 4.6, auto_width_height: bool = True) -> List:
    """
    TODO: describe function
    """
    if gw == 0 or gh == 0:
        distanses = find_distances(rect)
        h = (distanses[0]['d'] + distanses[2]['d']) / 2
        if auto_width_height:
            w = int(h * coef)
        else:
            w = (distanses[1]['d'] + distanses[3]['d']) / 2
    else:
        w, h = gw, gh
    return build_perspective(img, rect, int(w), int(h))


def reshape_points(target_points: List or np.ndarray, start_idx: int) -> List:
    """
    TODO: describe function
    """
    if start_idx > 0:
        part1 = target_points[:start_idx]
        part2 = target_points[start_idx:]
        target_points = np.concatenate((part2, part1))
    return target_points


def crop_number_plate_zones_from_images(images, images_points):
    zones = []
    image_ids = []
    for i, (image, image_points) in enumerate(zip(images, images_points)):
        image_zones = []
        for rect in image_points:
            zone = get_cv_zone_rgb(image, reshape_points(rect, 1))
            image_zones.append(zone)
        for zone in image_zones:
            zones.append(zone)
            image_ids.append(i)
    return zones, image_ids


def crop_image(image, target_box):
    x = int(min(target_box[0], target_box[2]))
    w = int(abs(target_box[2] - target_box[0]))
    y = int(min(target_box[1], target_box[3]))
    h = int(abs(target_box[3] - target_box[1]))

    image_part = image[y:y + h, x:x + w]
    return image_part, (x, w, y, h)


def minimum_bounding_rectangle(points: np.ndarray) -> np.ndarray:
    """
    Find the smallest bounding rectangle for a set of points.
    detail: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def get_y_by_matrix(matrix: np.ndarray, x: float) -> np.ndarray:
    """
    TODO: describe function
    """
    matrix_a = matrix[0]
    matrix_b = matrix[1]
    matrix_c = matrix[2]
    if matrix_b != 0:
        return (matrix_c - matrix_a * x) / matrix_b


def order_points_old(pts: np.ndarray):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    lp = np.argmin(s)

    # fix original code by Oleg Cherniy
    rp = lp + 2
    if rp > 3:
        rp = rp - 4
    rect[0] = pts[lp]
    rect[2] = pts[rp]
    pts_crop = [pts[idx] for idx in filter(lambda i: (i != lp) and (i != rp), range(len(pts)))]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # Определяется так.
    # Предположим, у нас есть 3 точки: А(х1,у1), Б(х2,у2), С(х3,у3).
    # Через точки А и Б проведена прямая. И нам надо определить, как расположена точка С относительно прямой АБ.
    # Для этого вычисляем значение:
    # D = (х3 - х1) * (у2 - у1) - (у3 - у1) * (х2 - х1)
    # - Если D = 0 - значит, точка С лежит на прямой АБ.
    # - Если D < 0 - значит, точка С лежит слева от прямой.
    # - Если D > 0 - значит, точка С лежит справа от прямой.
    x1 = rect[0][0]
    y1 = rect[0][1]
    x2 = rect[2][0]
    y2 = rect[2][1]
    x3 = pts_crop[0][0]
    y3 = pts_crop[0][1]
    d = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if d > 0:
        rect[1] = pts_crop[0]
        rect[3] = pts_crop[1]
    else:
        rect[1] = pts_crop[1]
        rect[3] = pts_crop[0]

    # return the ordered coordinates
    return rect



def fix_clockwise2(target_points: np.ndarray or List) -> np.ndarray:
    return order_points_old(np.array(target_points))


def find_min_x_idx(target_points: Union) -> int:
    """
    TODO: describe function
    """
    min_x_idx = 3
    for i in range(0, len(target_points)):
        if target_points[i][0] < target_points[min_x_idx][0]:
            min_x_idx = i
        if target_points[i][0] == target_points[min_x_idx][0] and target_points[i][1] < target_points[min_x_idx][1]:
            min_x_idx = i
    return min_x_idx


def detect_intersection(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    www.math.by/geometry/eqline.html
    xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(x, y)