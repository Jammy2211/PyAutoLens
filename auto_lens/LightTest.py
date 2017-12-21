import math
from scipy.integrate import quad

def func(r):
    return 3.0 * math.exp(-7.664 * (((r / 2.0) ** (1. / 4.0)) - 1))

def circle_func(r):
    return 2 * math.pi * r * func(r)

def ell_func(x, major_axis, minor_axis):
    r = x * (minor_axis/major_axis)
    return 2 * math.pi * r * func(x)

radius = 10.0

area1 = math.pi * radius ** 2
area_int1 = quad(circle_func, a=0.0, b=radius)[0]

print(area1, area_int1)

axis_ratio = 0.3

minor_axis = radius * axis_ratio
major_axis = radius

area = math.pi * major_axis * minor_axis
radius_int = major_axis
area_int = quad(ell_func, a=0.0, b=radius_int, args=(major_axis, minor_axis))[0]

print(area, area_int)

print(area1/area_int1)
print(area/area_int)