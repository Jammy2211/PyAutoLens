import cProfile
from profile import mass_profile
import pytest
from numba import cfunc, carray, jit
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer
import numpy as np
from scipy import ndimage as ndi
from scipy import LowLevelCallable
from scipy import integrate

invexp = lambda x: np.exp(-x)
integrate.quad(invexp, 0, np.inf)

image = np.random.random((2048, 2048))

footprint = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)


def test_deflection_angles():
    sersic = mass_profile.SersicMassProfile(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0, flux=5.0,
                                            effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)

    defls = sersic.deflections_at_coordinates(coordinates=(0.1625, 0.1625))

    assert defls[0] == pytest.approx(0.79374, 1e-3)
    assert defls[1] == pytest.approx(1.1446, 1e-3)

    for i in range(20):
        for j in range(20):
            sersic.deflections_at_coordinates(coordinates=(i * 0.01625, j * 0.01625))


# cProfile.run('test_deflection_angles()')


@jit
def other_func():
    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nbmin(values_ptr, len_values, result, data):
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.inf
    for v in values:

        if v < result[0]:
            result[0] = v
            other_func()

    return 1


# LowLevelCallable(nbmin.ctypes)()

print(ndi.generic_filter(image, LowLevelCallable(nbmin.ctypes), footprint=footprint))
