import numpy as np
import math


def translate_coordinates(x, y, x_cen, y_cen):
    return (x - x_cen), (y - y_cen)


def calc_radial_distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def rotate_coordinates(x, y, phi_degrees):
    phi_radians = math.radians(phi_degrees)
    cos_phi = math.cos(phi_radians)
    sin_phi = math.sin(phi_radians)

    r = calc_radial_distance(x, y)

    # 2D rotation matrix
    cos_theta = (x / r) * cos_phi + (y / r) * sin_phi
    sin_theta = (y / r) * cos_phi - (x / r) * sin_phi

    x_rot = r * cos_theta
    y_rot = r * sin_theta

    return x_rot, y_rot


def sie_defl_angle(x, y, x_cen, y_cen, ein_r, q, phi):
    x_shift, y_shift = translate_coordinates(x=x, y=y, x_cen=x_cen, y_cen=y_cen)
    x_rot, y_rot = rotate_coordinates(x=x_shift, y=y_shift, phi_degrees=phi)




    # class MassProfile(object):
    #
    #     def __init__(self):
    #
    #         return
    #
    # class SIS(MassProfile):
    #
    #     def __init__(self):
    #
    #         return
