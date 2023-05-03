"""
Functions and classes math and 3d geometry related
"""
import math
import sys

sys.path.append('/home/scasao/pytorch/multi-target_tracking/')


class Cylinder:
    def __init__(self, center, width, height):
        self.center = center
        self.width = width
        self.height = height

    @classmethod
    def XYWH(cls, x, y, w, h):
        return cls(Point3D(x, y, 0), w, h)

    @classmethod
    def XYZWH(cls, x, y, z, w, h):
        return cls(Point3D(x, y, z), w, h)

    def getCenter(self):
        return self.center

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getXYZWH(self):
        x, y, z = self.getCenter().getXYZ()
        return x, y, z, self.getWidth(), self.getHeight()

    def getXYWH(self):
        x, y, z = self.getCenter().getXYZ()
        return x, y, self.getWidth(), self.getHeight()

    def setCenter(self, center):
        self.center = center

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height

    def getFeet(self):
        x, y, z = self.getCenter().getXYZ()
        zFeet = z - 0.05 * z
        return Point3D(x, y, zFeet)

    def getHair(self):
        x, y, z = self.getCenter().getXYZ()
        zHair = z + 0.05 * self.height + self.height
        return Point3D(x, y, zHair)


class Point3D:
    """
    Represents a 3d point.
    """

    def __init__(self, x, y, z=0., s=1.):
        self.x = x
        self.y = y
        self.z = z
        self.s = s

    def getXYZ(self):
        return float(self.x) / self.s, float(self.y) / self.s, self.z

    def getAsXY(self):
        return float(self.x) / self.s, float(self.y) / self.s

    def normalize(self, dist=1.):
        return Point3D(self.x, self.y, self.z, math.sqrt(self.x ** 2 + self.y ** 2) / dist)


def f_euclidian_ground(a, b):
    """
    returns the euclidian distance between the two points
    """
    ax, ay, az = a.getXYZ()
    bx, by, az = b.getXYZ()
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def f_subtract_ground(a, b):
    """
    return difference of points a-b
    :param a:
    :param b:
    :return:
    """
    return Point3D(b.s * a.x - a.s * b.x, b.s * a.y - a.s * b.y, b.z, a.s * b.s)


def f_add_ground(a, b):
    """
    return addition of points a+b
    :param a:
    :param b:
    :return:
    """
    return Point3D(b.s * a.x + a.s * b.x, b.s * a.y + a.s * b.y, b.z, a.s * b.s)

