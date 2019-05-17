import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import acos
from math import sqrt
from math import pi

def length(v):
  return sqrt(dot_product(v, v))

def dot_product(v1, v2):
  return sum((a * b) for a, b in zip(v1, v2))

def determinant(v, w):
   return v[0] * w[1] - v[1] * w[0]

def inner_angle(v, w):
   cosx = dot_product(v, w) / (length(v) * length(w))
   rad = acos(cosx) # in radians
   return rad * 180 / pi # returns degrees

def angle_clockwise(A, B):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360 - inner

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def direction(v1, v2):
    result = {True: 1, False: -1}
    return result[angle_clockwise(v1, v2) > 180]

def calc_vectors_intersection(tracker_vector, ROI, line):
    line = (np.array(line[0]), np.array(line[1]))
    intersection = line_intersection(tracker_vector, line)
    if intersection is None:
        return None
    point_of_intersection = Point(intersection)
    polygon = Polygon(ROI)
    return polygon.contains(point_of_intersection), direction(tracker_vector[1] - tracker_vector[0], line[1] - line[0])

# if __name__ == '__main__':
#     calc_vectors_intersection((np.array([1,4]), np.array([7, 3])), ((0,0), (10, 0), (10, 10), (0, 10)), (np.array([9, 2]), np.array([9,7])))
