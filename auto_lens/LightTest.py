# from profile import light_profile
# import matplotlib.pyplot as plt
# import numpy as np
#
# sersic = light_profile.SersicLightProfile(axis_ratio=0.3, phi=80.0, flux=3.0,
#                                           effective_radius=2.0, sersic_index=4.0)
#
# xs = np.linspace(-0.5, 0.5, 400)
# ys = np.linspace(-0.5, 0.5, 400)
#
# edge = xs[1] - xs[0]
# area = edge ** 2
#
# flux_grid = np.zeros((400, 400))
#
# for ix, x in enumerate(xs):
#     for iy, y in enumerate(ys):
#         flux_grid[ix, iy] = sersic.flux_at_coordinates((x, y))
#
# plt.imshow(flux_grid)
# plt.show()

import math
from scipy.integrate import quad

def func(r):
    return r**2

def circle_func(r):
    return 2 * math.pi * r * func(r)

def ell_func(r, a, b):
    return 4 * b * math.sqrt(1 - ((r**2)/(a**2)) ) * func(r)

def ell_func2(r, a, b):
    return 4 * (b / a) * math.sqrt(a**2 - r**2) * func(r*math.pi/2)

radius = 5.0

area = math.pi * radius ** 2
area_int = quad(circle_func, a=0.0, b=radius)[0]

print(area, area_int)

b = 5.0
a = radius

q = b/a

area = math.pi * b * a
radius_int = a
area_int = quad(ell_func2, a=0.0, b=radius_int, args=(a, b))[0]

print(area, area_int)