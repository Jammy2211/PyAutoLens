import numpy as np
import pytest

from autolens.data.array import interpolation, mask
from autolens.model.galaxy import galaxy
from autolens.model.profiles import mass_profiles


@pytest.fixture(name='scheme')
def make_scheme():
    return interpolation.InterpolationScheme(shape=(3, 3), image_coords=np.array([[1.0, 1.0]]), image_pixel_scale=1.0)


@pytest.fixture(name='geometry')
def make_geometry():
    return interpolation.InterpolationGeometry(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0,
                                               y_pixel_scale=1.0, x_pixel_scale=1.0)


@pytest.fixture(name='galaxy_no_profiles', scope='function')
def make_galaxy_no_profiles():
    return galaxy.Galaxy()


@pytest.fixture(name="galaxy_mass_sis")
def make_galaxy_mass_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile=sis)


class TestInterpolationScheme(object):
    class TestConstructor:

        def test__sets_up_attributes_correctly(self):
            image_coords = np.array([[-1.0, -6.0], [-1.0, 0.0], [-4.0, 2.0],
                                     [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                     [3.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=image_coords, image_pixel_scale=1.0)

            assert interp.shape == (3, 3)
            assert interp.pixels == 9
            assert (interp.image_coords == image_coords).all()
            assert interp.geometry.y_min == -6.0
            assert interp.geometry.y_max == 2.0
            assert interp.geometry.x_min == -4.0
            assert interp.geometry.x_max == 3.0
            assert interp.geometry.y_pixel_scale == 1.0
            assert interp.geometry.x_pixel_scale == 1.0
            assert interp.geometry.x_size == 7.0
            assert interp.geometry.y_size == 8.0
            assert interp.geometry.x_start == -4.5
            assert interp.geometry.y_start == -6.5

    class TestNeighbors:

        def test___3x3_grid_neighbors_all_correct(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=np.array([[1.0, 1.0]]),
                                                       image_pixel_scale=1.0)

            assert (interp.bottom_right_neighbors[0] == np.array([1, 3, 4])).all()
            assert (interp.bottom_right_neighbors[1] == np.array([2, 4, 5])).all()
            assert (interp.bottom_right_neighbors[2] == np.array([-1, 5, -1])).all()
            assert (interp.bottom_right_neighbors[3] == np.array([4, 6, 7])).all()
            assert (interp.bottom_right_neighbors[4] == np.array([5, 7, 8])).all()
            assert (interp.bottom_right_neighbors[5] == np.array([-1, 8, -1])).all()
            assert (interp.bottom_right_neighbors[6] == np.array([7, -1, -1])).all()
            assert (interp.bottom_right_neighbors[7] == np.array([8, -1, -1])).all()
            assert (interp.bottom_right_neighbors[8] == np.array([-1, -1, -1])).all()

            assert (interp.bottom_left_neighbors[0] == np.array([-1, -1, 3])).all()
            assert (interp.bottom_left_neighbors[1] == np.array([0, 3, 4])).all()
            assert (interp.bottom_left_neighbors[2] == np.array([1, 4, 5])).all()
            assert (interp.bottom_left_neighbors[3] == np.array([-1, -1, 6])).all()
            assert (interp.bottom_left_neighbors[4] == np.array([3, 6, 7])).all()
            assert (interp.bottom_left_neighbors[5] == np.array([4, 7, 8])).all()
            assert (interp.bottom_left_neighbors[6] == np.array([-1, -1, -1])).all()
            assert (interp.bottom_left_neighbors[7] == np.array([6, -1, -1])).all()
            assert (interp.bottom_left_neighbors[8] == np.array([7, -1, -1])).all()

            assert (interp.top_right_neighbors[0] == np.array([-1, -1, 1])).all()
            assert (interp.top_right_neighbors[1] == np.array([-1, -1, 2])).all()
            assert (interp.top_right_neighbors[2] == np.array([-1, -1, -1])).all()
            assert (interp.top_right_neighbors[3] == np.array([0, 1, 4])).all()
            assert (interp.top_right_neighbors[4] == np.array([1, 2, 5])).all()
            assert (interp.top_right_neighbors[5] == np.array([2, -1, -1])).all()
            assert (interp.top_right_neighbors[6] == np.array([3, 4, 7])).all()
            assert (interp.top_right_neighbors[7] == np.array([4, 5, 8])).all()
            assert (interp.top_right_neighbors[8] == np.array([5, -1, -1])).all()

            assert (interp.top_left_neighbors[0] == np.array([-1, -1, -1])).all()
            assert (interp.top_left_neighbors[1] == np.array([-1, -1, 0])).all()
            assert (interp.top_left_neighbors[2] == np.array([-1, -1, 1])).all()
            assert (interp.top_left_neighbors[3] == np.array([-1, 0, -1])).all()
            assert (interp.top_left_neighbors[4] == np.array([0, 1, 3])).all()
            assert (interp.top_left_neighbors[5] == np.array([1, 2, 4])).all()
            assert (interp.top_left_neighbors[6] == np.array([-1, 3, -1])).all()
            assert (interp.top_left_neighbors[7] == np.array([3, 4, 6])).all()
            assert (interp.top_left_neighbors[8] == np.array([4, 5, 7])).all()

        def test___3x4_grid_neighbors_all_correct(self):
            # |0|1| 2| 3|
            # |4|5| 6| 7|
            # |8|9|10|11|

            interp = interpolation.InterpolationScheme(shape=(3, 4), image_coords=np.array([[1.0, 1.0]]),
                                                       image_pixel_scale=1.0)

            assert (interp.bottom_right_neighbors[0] == np.array([1, 4, 5])).all()
            assert (interp.bottom_right_neighbors[1] == np.array([2, 5, 6])).all()
            assert (interp.bottom_right_neighbors[2] == np.array([3, 6, 7])).all()
            assert (interp.bottom_right_neighbors[3] == np.array([-1, 7, -1])).all()
            assert (interp.bottom_right_neighbors[4] == np.array([5, 8, 9])).all()
            assert (interp.bottom_right_neighbors[5] == np.array([6, 9, 10])).all()
            assert (interp.bottom_right_neighbors[6] == np.array([7, 10, 11])).all()
            assert (interp.bottom_right_neighbors[7] == np.array([-1, 11, -1])).all()
            assert (interp.bottom_right_neighbors[8] == np.array([9, -1, -1])).all()
            assert (interp.bottom_right_neighbors[9] == np.array([10, -1, -1])).all()
            assert (interp.bottom_right_neighbors[10] == np.array([11, -1, -1])).all()
            assert (interp.bottom_right_neighbors[11] == np.array([-1, -1, -1])).all()

            assert (interp.bottom_left_neighbors[0] == np.array([-1, -1, 4])).all()
            assert (interp.bottom_left_neighbors[1] == np.array([0, 4, 5])).all()
            assert (interp.bottom_left_neighbors[2] == np.array([1, 5, 6])).all()
            assert (interp.bottom_left_neighbors[3] == np.array([2, 6, 7])).all()
            assert (interp.bottom_left_neighbors[4] == np.array([-1, -1, 8])).all()
            assert (interp.bottom_left_neighbors[5] == np.array([4, 8, 9])).all()
            assert (interp.bottom_left_neighbors[6] == np.array([5, 9, 10])).all()
            assert (interp.bottom_left_neighbors[7] == np.array([6, 10, 11])).all()
            assert (interp.bottom_left_neighbors[8] == np.array([-1, -1, -1])).all()
            assert (interp.bottom_left_neighbors[9] == np.array([8, -1, -1])).all()
            assert (interp.bottom_left_neighbors[10] == np.array([9, -1, -1])).all()
            assert (interp.bottom_left_neighbors[11] == np.array([10, -1, -1])).all()

            assert (interp.top_right_neighbors[0] == np.array([-1, -1, 1])).all()
            assert (interp.top_right_neighbors[1] == np.array([-1, -1, 2])).all()
            assert (interp.top_right_neighbors[2] == np.array([-1, -1, 3])).all()
            assert (interp.top_right_neighbors[3] == np.array([-1, -1, -1])).all()
            assert (interp.top_right_neighbors[4] == np.array([0, 1, 5])).all()
            assert (interp.top_right_neighbors[5] == np.array([1, 2, 6])).all()
            assert (interp.top_right_neighbors[6] == np.array([2, 3, 7])).all()
            assert (interp.top_right_neighbors[7] == np.array([3, -1, -1])).all()
            assert (interp.top_right_neighbors[8] == np.array([4, 5, 9])).all()
            assert (interp.top_right_neighbors[9] == np.array([5, 6, 10])).all()
            assert (interp.top_right_neighbors[10] == np.array([6, 7, 11])).all()
            assert (interp.top_right_neighbors[11] == np.array([7, -1, -1])).all()

            assert (interp.top_left_neighbors[0] == np.array([-1, -1, -1])).all()
            assert (interp.top_left_neighbors[1] == np.array([-1, -1, 0])).all()
            assert (interp.top_left_neighbors[2] == np.array([-1, -1, 1])).all()
            assert (interp.top_left_neighbors[3] == np.array([-1, -1, 2])).all()
            assert (interp.top_left_neighbors[4] == np.array([-1, 0, -1])).all()
            assert (interp.top_left_neighbors[5] == np.array([0, 1, 4])).all()
            assert (interp.top_left_neighbors[6] == np.array([1, 2, 5])).all()
            assert (interp.top_left_neighbors[7] == np.array([2, 3, 6])).all()
            assert (interp.top_left_neighbors[8] == np.array([-1, 4, -1])).all()
            assert (interp.top_left_neighbors[9] == np.array([4, 5, 8])).all()
            assert (interp.top_left_neighbors[10] == np.array([5, 6, 9])).all()
            assert (interp.top_left_neighbors[11] == np.array([6, 7, 10])).all()

        def test___4x3_grid_neighbors_all_correct(self):
            # |0| 1| 2|
            # |3| 4| 5|
            # |6| 7| 8|
            # |9|10|11|

            interp = interpolation.InterpolationScheme(shape=(4, 3), image_coords=np.array([[1.0, 1.0]]),
                                                       image_pixel_scale=1.0)

            assert (interp.bottom_right_neighbors[0] == np.array([1, 3, 4])).all()
            assert (interp.bottom_right_neighbors[1] == np.array([2, 4, 5])).all()
            assert (interp.bottom_right_neighbors[2] == np.array([-1, 5, -1])).all()
            assert (interp.bottom_right_neighbors[3] == np.array([4, 6, 7])).all()
            assert (interp.bottom_right_neighbors[4] == np.array([5, 7, 8])).all()
            assert (interp.bottom_right_neighbors[5] == np.array([-1, 8, -1])).all()
            assert (interp.bottom_right_neighbors[6] == np.array([7, 9, 10])).all()
            assert (interp.bottom_right_neighbors[7] == np.array([8, 10, 11])).all()
            assert (interp.bottom_right_neighbors[8] == np.array([-1, 11, -1])).all()
            assert (interp.bottom_right_neighbors[9] == np.array([10, -1, -1])).all()
            assert (interp.bottom_right_neighbors[10] == np.array([11, -1, -1])).all()
            assert (interp.bottom_right_neighbors[11] == np.array([-1, -1, -1])).all()

            assert (interp.bottom_left_neighbors[0] == np.array([-1, -1, 3])).all()
            assert (interp.bottom_left_neighbors[1] == np.array([0, 3, 4])).all()
            assert (interp.bottom_left_neighbors[2] == np.array([1, 4, 5])).all()
            assert (interp.bottom_left_neighbors[3] == np.array([-1, -1, 6])).all()
            assert (interp.bottom_left_neighbors[4] == np.array([3, 6, 7])).all()
            assert (interp.bottom_left_neighbors[5] == np.array([4, 7, 8])).all()
            assert (interp.bottom_left_neighbors[6] == np.array([-1, -1, 9])).all()
            assert (interp.bottom_left_neighbors[7] == np.array([6, 9, 10])).all()
            assert (interp.bottom_left_neighbors[8] == np.array([7, 10, 11])).all()
            assert (interp.bottom_left_neighbors[9] == np.array([-1, -1, -1])).all()
            assert (interp.bottom_left_neighbors[10] == np.array([9, -1, -1])).all()
            assert (interp.bottom_left_neighbors[11] == np.array([10, -1, -1])).all()

            assert (interp.top_right_neighbors[0] == np.array([-1, -1, 1])).all()
            assert (interp.top_right_neighbors[1] == np.array([-1, -1, 2])).all()
            assert (interp.top_right_neighbors[2] == np.array([-1, -1, -1])).all()
            assert (interp.top_right_neighbors[3] == np.array([0, 1, 4])).all()
            assert (interp.top_right_neighbors[4] == np.array([1, 2, 5])).all()
            assert (interp.top_right_neighbors[5] == np.array([2, -1, -1])).all()
            assert (interp.top_right_neighbors[6] == np.array([3, 4, 7])).all()
            assert (interp.top_right_neighbors[7] == np.array([4, 5, 8])).all()
            assert (interp.top_right_neighbors[8] == np.array([5, -1, -1])).all()
            assert (interp.top_right_neighbors[9] == np.array([6, 7, 10])).all()
            assert (interp.top_right_neighbors[10] == np.array([7, 8, 11])).all()
            assert (interp.top_right_neighbors[11] == np.array([8, -1, -1])).all()

            assert (interp.top_left_neighbors[0] == np.array([-1, -1, -1])).all()
            assert (interp.top_left_neighbors[1] == np.array([-1, -1, 0])).all()
            assert (interp.top_left_neighbors[2] == np.array([-1, -1, 1])).all()
            assert (interp.top_left_neighbors[3] == np.array([-1, 0, -1])).all()
            assert (interp.top_left_neighbors[4] == np.array([0, 1, 3])).all()
            assert (interp.top_left_neighbors[5] == np.array([1, 2, 4])).all()
            assert (interp.top_left_neighbors[6] == np.array([-1, 3, -1])).all()
            assert (interp.top_left_neighbors[7] == np.array([3, 4, 6])).all()
            assert (interp.top_left_neighbors[8] == np.array([4, 5, 7])).all()
            assert (interp.top_left_neighbors[9] == np.array([-1, 6, -1])).all()
            assert (interp.top_left_neighbors[10] == np.array([6, 7, 9])).all()
            assert (interp.top_left_neighbors[11] == np.array([7, 8, 10])).all()

        def test___4x4_grid_neighbors_all_correct(self):
            # | 0| 1| 2| 3|
            # | 4| 5| 6| 7|
            # | 8| 9|10|11|
            # |12|13|14|15|

            interp = interpolation.InterpolationScheme(shape=(4, 4), image_coords=np.array([[1.0, 1.0]]),
                                                       image_pixel_scale=1.0)

            assert (interp.bottom_right_neighbors[0] == np.array([1, 4, 5])).all()
            assert (interp.bottom_right_neighbors[1] == np.array([2, 5, 6])).all()
            assert (interp.bottom_right_neighbors[2] == np.array([3, 6, 7])).all()
            assert (interp.bottom_right_neighbors[3] == np.array([-1, 7, -1])).all()
            assert (interp.bottom_right_neighbors[4] == np.array([5, 8, 9])).all()
            assert (interp.bottom_right_neighbors[5] == np.array([6, 9, 10])).all()
            assert (interp.bottom_right_neighbors[6] == np.array([7, 10, 11])).all()
            assert (interp.bottom_right_neighbors[7] == np.array([-1, 11, -1])).all()
            assert (interp.bottom_right_neighbors[8] == np.array([9, 12, 13])).all()
            assert (interp.bottom_right_neighbors[9] == np.array([10, 13, 14])).all()
            assert (interp.bottom_right_neighbors[10] == np.array([11, 14, 15])).all()
            assert (interp.bottom_right_neighbors[11] == np.array([-1, 15, -1])).all()
            assert (interp.bottom_right_neighbors[12] == np.array([13, -1, -1])).all()
            assert (interp.bottom_right_neighbors[13] == np.array([14, -1, -1])).all()
            assert (interp.bottom_right_neighbors[14] == np.array([15, -1, -1])).all()
            assert (interp.bottom_right_neighbors[15] == np.array([-1, -1, -1])).all()

            assert (interp.bottom_left_neighbors[0] == np.array([-1, -1, 4])).all()
            assert (interp.bottom_left_neighbors[1] == np.array([0, 4, 5])).all()
            assert (interp.bottom_left_neighbors[2] == np.array([1, 5, 6])).all()
            assert (interp.bottom_left_neighbors[3] == np.array([2, 6, 7])).all()
            assert (interp.bottom_left_neighbors[4] == np.array([-1, -1, 8])).all()
            assert (interp.bottom_left_neighbors[5] == np.array([4, 8, 9])).all()
            assert (interp.bottom_left_neighbors[6] == np.array([5, 9, 10])).all()
            assert (interp.bottom_left_neighbors[7] == np.array([6, 10, 11])).all()
            assert (interp.bottom_left_neighbors[8] == np.array([-1, -1, 12])).all()
            assert (interp.bottom_left_neighbors[9] == np.array([8, 12, 13])).all()
            assert (interp.bottom_left_neighbors[10] == np.array([9, 13, 14])).all()
            assert (interp.bottom_left_neighbors[11] == np.array([10, 14, 15])).all()
            assert (interp.bottom_left_neighbors[12] == np.array([-1, -1, -1])).all()
            assert (interp.bottom_left_neighbors[13] == np.array([12, -1, -1])).all()
            assert (interp.bottom_left_neighbors[14] == np.array([13, -1, -1])).all()
            assert (interp.bottom_left_neighbors[15] == np.array([14, -1, -1])).all()

            assert (interp.top_right_neighbors[0] == np.array([-1, -1, 1])).all()
            assert (interp.top_right_neighbors[1] == np.array([-1, -1, 2])).all()
            assert (interp.top_right_neighbors[2] == np.array([-1, -1, 3])).all()
            assert (interp.top_right_neighbors[3] == np.array([-1, -1, -1])).all()
            assert (interp.top_right_neighbors[4] == np.array([0, 1, 5])).all()
            assert (interp.top_right_neighbors[5] == np.array([1, 2, 6])).all()
            assert (interp.top_right_neighbors[6] == np.array([2, 3, 7])).all()
            assert (interp.top_right_neighbors[7] == np.array([3, -1, -1])).all()
            assert (interp.top_right_neighbors[8] == np.array([4, 5, 9])).all()
            assert (interp.top_right_neighbors[9] == np.array([5, 6, 10])).all()
            assert (interp.top_right_neighbors[10] == np.array([6, 7, 11])).all()
            assert (interp.top_right_neighbors[11] == np.array([7, -1, -1])).all()
            assert (interp.top_right_neighbors[12] == np.array([8, 9, 13])).all()
            assert (interp.top_right_neighbors[13] == np.array([9, 10, 14])).all()
            assert (interp.top_right_neighbors[14] == np.array([10, 11, 15])).all()
            assert (interp.top_right_neighbors[15] == np.array([11, -1, -1])).all()

            assert (interp.top_left_neighbors[0] == np.array([-1, -1, -1])).all()
            assert (interp.top_left_neighbors[1] == np.array([-1, -1, 0])).all()
            assert (interp.top_left_neighbors[2] == np.array([-1, -1, 1])).all()
            assert (interp.top_left_neighbors[3] == np.array([-1, -1, 2])).all()
            assert (interp.top_left_neighbors[4] == np.array([-1, 0, -1])).all()
            assert (interp.top_left_neighbors[5] == np.array([0, 1, 4])).all()
            assert (interp.top_left_neighbors[6] == np.array([1, 2, 5])).all()
            assert (interp.top_left_neighbors[7] == np.array([2, 3, 6])).all()
            assert (interp.top_left_neighbors[8] == np.array([-1, 4, -1])).all()
            assert (interp.top_left_neighbors[9] == np.array([4, 5, 8])).all()
            assert (interp.top_left_neighbors[10] == np.array([5, 6, 9])).all()
            assert (interp.top_left_neighbors[11] == np.array([6, 7, 10])).all()
            assert (interp.top_left_neighbors[12] == np.array([-1, 8, -1])).all()
            assert (interp.top_left_neighbors[13] == np.array([8, 9, 12])).all()
            assert (interp.top_left_neighbors[14] == np.array([9, 10, 13])).all()
            assert (interp.top_left_neighbors[15] == np.array([10, 11, 14])).all()

    class TestFromMask:

        def test__passes_mask_pixel_scale(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_pixel_scale == msk.pixel_scale

        def test__3x3_mask_with_1_pixel__3x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.0, 1.0]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([0.0, -1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, 1.0]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([1.0, -1.0]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.0, 1.0]), 1e-4)

        def test__same_as_above__change_pixel_scale(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-2.0, -2.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-2.0, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-2.0, 2.0]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([0.0, -2.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, 2.0]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([2.0, -2.0]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([2.0, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([2.0, 2.0]), 1e-4)

        def test__3x3_mask_with_1_pixel__4x4_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(4, 4))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([-1.0, 1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([-(1. / 3.), -1.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([-(1. / 3.), -(1. / 3.)]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([-(1. / 3.), (1. / 3.)]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([-(1. / 3.), 1.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([(1. / 3.), -1.0]), 1e-4)
            assert interp.image_coords[9] == pytest.approx(np.array([(1. / 3.), -(1. / 3.)]), 1e-4)
            assert interp.image_coords[10] == pytest.approx(np.array([(1. / 3.), (1. / 3.)]), 1e-4)
            assert interp.image_coords[11] == pytest.approx(np.array([(1. / 3.), 1.0]), 1e-4)
            assert interp.image_coords[12] == pytest.approx(np.array([1.0, -1.0]), 1e-4)
            assert interp.image_coords[13] == pytest.approx(np.array([1.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[14] == pytest.approx(np.array([1.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[15] == pytest.approx(np.array([1.0, 1.0]), 1e-4)

        def test__3x3_mask_with_1_pixel__3x4_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 4))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([-1.0, 1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, -1.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([0.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([0.0, 1.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.0, -1.0]), 1e-4)
            assert interp.image_coords[9] == pytest.approx(np.array([1.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[10] == pytest.approx(np.array([1.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[11] == pytest.approx(np.array([1.0, 1.0]), 1e-4)

        def test__3x3_mask_with_1_pixel__4x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(4, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.0, 1.0]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([-(1. / 3.), -1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([-(1. / 3.), 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([-(1. / 3.), 1.0]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([(1. / 3.), -1.0]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([(1. / 3.), 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([(1. / 3.), 1.0]), 1e-4)
            assert interp.image_coords[9] == pytest.approx(np.array([1.0, -1.0]), 1e-4)
            assert interp.image_coords[10] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert interp.image_coords[11] == pytest.approx(np.array([1.0, 1.0]), 1e-4)

        def test__4x4_mask_with_4_pixels__3x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, False, False, True],
                            [True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.5, -1.5]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.5, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.5, 1.5]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([0.0, -1.5]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, 1.5]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([1.5, -1.5]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([1.5, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.5, 1.5]), 1e-4)

        def test__3x4_mask_with_2_pixels__3x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.5, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.5, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.5, 1.0]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([0.0, -1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, 1.0]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([1.5, -1.0]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([1.5, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.5, 1.0]), 1e-4)

        def test__4x3_mask_with_4_pixels__3x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.0, -1.5]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.0, 1.5]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([0.0, -1.5]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, 1.5]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([1.0, -1.5]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.0, 1.5]), 1e-4)

        def test__3x4_mask_with_2_pixels__3x4_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(3, 4))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.5, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.5, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.5, (1. / 3.)]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([-1.5, 1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([0.0, -1.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([0.0, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([0.0, (1. / 3.)]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([0.0, 1.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([1.5, -1.0]), 1e-4)
            assert interp.image_coords[9] == pytest.approx(np.array([1.5, -(1. / 3.)]), 1e-4)
            assert interp.image_coords[10] == pytest.approx(np.array([1.5, (1. / 3.)]), 1e-4)
            assert interp.image_coords[11] == pytest.approx(np.array([1.5, 1.0]), 1e-4)

        def test__4x3_mask_with_2_pixels__4x3_interp_grid__image_coords_extend_beyond_mask(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            interp = interpolation.InterpolationScheme.from_mask(mask=msk, shape=(4, 3))

            assert interp.image_coords[0] == pytest.approx(np.array([-1.5, -1.0]), 1e-4)
            assert interp.image_coords[1] == pytest.approx(np.array([-1.5, 0.0]), 1e-4)
            assert interp.image_coords[2] == pytest.approx(np.array([-1.5, 1.0]), 1e-4)
            assert interp.image_coords[3] == pytest.approx(np.array([-0.5, -1.0]), 1e-4)
            assert interp.image_coords[4] == pytest.approx(np.array([-0.5, 0.0]), 1e-4)
            assert interp.image_coords[5] == pytest.approx(np.array([-0.5, 1.0]), 1e-4)
            assert interp.image_coords[6] == pytest.approx(np.array([0.5, -1.0]), 1e-4)
            assert interp.image_coords[7] == pytest.approx(np.array([0.5, 0.0]), 1e-4)
            assert interp.image_coords[8] == pytest.approx(np.array([0.5, 1.0]), 1e-4)
            assert interp.image_coords[9] == pytest.approx(np.array([1.5, -1.0]), 1e-4)
            assert interp.image_coords[10] == pytest.approx(np.array([1.5, 0.0]), 1e-4)
            assert interp.image_coords[11] == pytest.approx(np.array([1.5, 1.0]), 1e-4)

    class TestInterpolationCoordinatesFromSizes:

        def test__x_and_y_are_same_sizes__scales_are_1__new_coordinates_are_image_coordinates(self):
            image_coords = np.array([[-1.0, -6.0], [-1.0, 0.0], [-4.0, 2.0],
                                     [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                     [3.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=image_coords, image_pixel_scale=1.0)

            interp_coords = interp.interpolation_coordinates_from_sizes(new_x_size=7.0, new_y_size=8.0)

            assert (interp_coords == image_coords).all()
            assert interp_coords.geometry.y_min == -6.0
            assert interp_coords.geometry.y_max == 2.0
            assert interp_coords.geometry.x_min == -4.0
            assert interp_coords.geometry.x_max == 3.0
            assert interp_coords.geometry.y_size == 8.0
            assert interp_coords.geometry.x_size == 7.0
            assert interp_coords.geometry.y_pixel_scale == 1.0
            assert interp_coords.geometry.x_pixel_scale == 1.0
            assert interp_coords.geometry.x_start == -4.5
            assert interp_coords.geometry.y_start == -6.5
            assert interp_coords.scheme == interp

        def test__same_as_above_but_x_dimension_halves(self):
            image_coords = np.array([[-1.0, -6.0], [-1.0, 0.0], [-4.0, 2.0],
                                     [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                     [3.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=image_coords, image_pixel_scale=1.0)

            interp_coords = interp.interpolation_coordinates_from_sizes(new_x_size=3.5, new_y_size=8.0)

            assert (interp_coords == np.array([[-0.5, -6.0], [-0.5, 0.0], [-2.0, 2.0],
                                               [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                               [1.5, -1.0], [0.5, 0.0], [0.5, 1.0]])).all()
            assert interp_coords.geometry.y_min == -6.0
            assert interp_coords.geometry.y_max == 2.0
            assert interp_coords.geometry.x_min == -2.0
            assert interp_coords.geometry.x_max == 1.5
            assert interp_coords.geometry.y_size == 8.0
            assert interp_coords.geometry.x_size == 3.5
            assert interp_coords.geometry.y_pixel_scale == 1.0
            assert interp_coords.geometry.x_pixel_scale == 0.5
            assert interp_coords.geometry.x_start == -2.25
            assert interp_coords.geometry.y_start == -6.5
            assert (interp_coords.scheme == interp)

        def test__same_as_above_but_y_dimension_halves(self):
            image_coords = np.array([[-1.0, -6.0], [-1.0, 0.0], [-4.0, 2.0],
                                     [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                     [3.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=image_coords, image_pixel_scale=1.0)

            interp_coords = interp.interpolation_coordinates_from_sizes(new_x_size=7.0, new_y_size=4.0)

            assert (interp_coords == np.array([[-1.0, -3.0], [-1.0, 0.0], [-4.0, 1.0],
                                               [-0.0, -0.5], [0.0, 0.0], [0.0, 0.5],
                                               [3.0, -0.5], [1.0, 0.0], [1.0, 0.5]])).all()
            assert interp_coords.geometry.y_min == -3.0
            assert interp_coords.geometry.y_max == 1.0
            assert interp_coords.geometry.x_min == -4.0
            assert interp_coords.geometry.x_max == 3.0
            assert interp_coords.geometry.y_size == 4.0
            assert interp_coords.geometry.x_size == 7.0
            assert interp_coords.geometry.y_pixel_scale == 0.5
            assert interp_coords.geometry.x_pixel_scale == 1.0
            assert interp_coords.geometry.y_start == -3.25
            assert interp_coords.geometry.x_start == -4.5
            assert (interp_coords.scheme == interp)

        def test__same_as_above_triple_both_dimensions(self):
            image_coords = np.array([[-1.0, -6.0], [-1.0, 0.0], [-4.0, 2.0],
                                     [-0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                     [3.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            interp = interpolation.InterpolationScheme(shape=(3, 3), image_coords=image_coords, image_pixel_scale=1.0)

            interp_coords = interp.interpolation_coordinates_from_sizes(new_x_size=21.0, new_y_size=24.0)

            assert (interp_coords == np.array([[-3.0, -18.0], [-3.0, 0.0], [-12.0, 6.0],
                                               [-0.0, -3.0], [0.0, 0.0], [0.0, 3.0],
                                               [9.0, -3.0], [3.0, 0.0], [3.0, 3.0]])).all()
            assert interp_coords.geometry.y_min == -18.0
            assert interp_coords.geometry.y_max == 6.0
            assert interp_coords.geometry.x_min == -12.0
            assert interp_coords.geometry.x_max == 9.0
            assert interp_coords.geometry.y_size == 24.0
            assert interp_coords.geometry.x_size == 21.0
            assert interp_coords.geometry.y_pixel_scale == 3.0
            assert interp_coords.geometry.x_pixel_scale == 3.0
            assert interp_coords.geometry.x_start == -13.5
            assert interp_coords.geometry.y_start == -19.5
            assert (interp_coords.scheme == interp)


class TestInterpolationCoordinates(object):
    class TestConstructor:

        def test__sets_up_coordinates_correctly(self):
            geometry = interpolation.InterpolationGeometry(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0,
                                                           y_pixel_scale=1.0, x_pixel_scale=1.0)

            scheme = interpolation.InterpolationScheme(shape=(3, 3), image_coords=np.array([[1.0, 1.0]]),
                                                       image_pixel_scale=1.0)

            interp_coords = interpolation.InterpolationCoordinates(array=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                                                                   geometry=geometry, scheme=scheme)

            assert (interp_coords == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])).all()
            assert interp_coords.geometry.y_min == -1.0
            assert interp_coords.geometry.y_max == 1.0
            assert interp_coords.geometry.x_min == -1.0
            assert interp_coords.geometry.x_max == 1.0
            assert interp_coords.geometry.y_size == 2.0
            assert interp_coords.geometry.x_size == 2.0
            assert interp_coords.geometry.y_pixel_scale == 1.0
            assert interp_coords.geometry.x_pixel_scale == 1.0
            assert interp_coords.geometry.y_start == -1.5
            assert interp_coords.geometry.x_start == -1.5
            assert interp_coords.scheme.shape == (3, 3)
            assert (interp_coords.scheme.image_coords == np.array([[1.0, 1.0]])).all()

    class TestInterpolationDeflectionsFromCoordinates:

        def test__galaxy_has_no_mass_profile__deflection_coordinates_are_0s(self, galaxy_no_profiles, geometry, scheme):
            interp_coords = interpolation.InterpolationCoordinates(array=np.array([[1.0, 1.0], [1.0, 0.0]]),
                                                                   geometry=geometry, scheme=scheme)

            interp_defls = interp_coords.interpolation_deflections_from_coordinates_and_galaxies(
                galaxies=[galaxy_no_profiles])

            assert (interp_defls == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (interp_defls.interp_coords == np.array([[1.0, 1.0], [1.0, 0.0]])).all()
            assert interp_defls.geometry == geometry
            assert interp_defls.scheme == scheme

        def test__galaxy_has_sis_mass_profile__deflection_coordinates_are_correct(self, galaxy_mass_sis, geometry,
                                                                                  scheme):
            interp_coords = interpolation.InterpolationCoordinates(array=np.array([[1.0, 1.0], [1.0, 0.0]]),
                                                                   geometry=geometry, scheme=scheme)

            interp_defls = interp_coords.interpolation_deflections_from_coordinates_and_galaxies(
                galaxies=[galaxy_mass_sis])

            assert interp_defls[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert interp_defls[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert (interp_defls.interp_coords == np.array([[1.0, 1.0], [1.0, 0.0]])).all()
            assert interp_defls.geometry == geometry
            assert interp_defls.scheme == scheme

        def test__x3_galaxies_with_sis_mass_profile__deflection_coordinates_are_correct(self, galaxy_mass_sis, geometry,
                                                                                        scheme):
            interp_coords = interpolation.InterpolationCoordinates(array=np.array([[1.0, 1.0], [1.0, 0.0]]),
                                                                   geometry=geometry, scheme=scheme)

            interp_defls = interp_coords.interpolation_deflections_from_coordinates_and_galaxies(
                galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert interp_defls[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert interp_defls[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert (interp_defls.interp_coords == np.array([[1.0, 1.0], [1.0, 0.0]])).all()
            assert interp_defls.geometry == geometry
            assert interp_defls.scheme == scheme


class TestInterpolationDeflections(object):
    class TestConstructor:

        def test__sets_up_attributes_correctly(self, geometry, scheme):
            interp_defls = interpolation.InterpolationDeflections(array=np.array([[1.0, 1.0]]),
                                                                  coords=np.array([[2.0, 2.0]]), geometry=geometry,
                                                                  scheme=scheme)

            assert (interp_defls == np.array([[1.0, 1.0]])).all()
            assert (interp_defls.interp_coords == np.array([[2.0, 2.0]])).all()
            assert interp_defls.geometry == geometry
            assert interp_defls.scheme == scheme

    class TestGridToInterp:

        def test__3x3_interp_scheme__coordinate_on_centre_of_each_pixel__grid_retuns_correct_index(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            # interp defls arn't ussed for grid to interp

            interp_defls = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

            interp_coords = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                      [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                      [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = interpolation.InterpolationGeometry(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0,
                                                           y_pixel_scale=1.0, x_pixel_scale=1.0)
            scheme = interpolation.InterpolationScheme(shape=(3, 3), image_coords=interp_coords, image_pixel_scale=1.0)

            interp_defls = interpolation.InterpolationDeflections(array=interp_defls, coords=interp_coords,
                                                                  geometry=geometry, scheme=scheme)

            grid_to_interp = interp_defls.grid_to_interp_from_grid(
                grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                               [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

            assert (grid_to_interp[0] == np.array([0])).all()
            assert (grid_to_interp[1] == np.array([1])).all()
            assert (grid_to_interp[2] == np.array([2])).all()
            assert (grid_to_interp[3] == np.array([3])).all()
            assert (grid_to_interp[4] == np.array([4])).all()
            assert (grid_to_interp[5] == np.array([5])).all()
            assert (grid_to_interp[6] == np.array([6])).all()
            assert (grid_to_interp[7] == np.array([7])).all()
            assert (grid_to_interp[8] == np.array([8])).all()

        def test__3x3_interp_scheme__coordinates_near_edges_of_pixels__different_ordering(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            # interp defls arn't ussed for bilinear interpolation weights

            interp_defls = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

            interp_coords = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                      [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                      [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = interpolation.InterpolationGeometry(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0,
                                                           y_pixel_scale=1.0, x_pixel_scale=1.0)
            scheme = interpolation.InterpolationScheme(shape=(3, 3), image_coords=interp_coords, image_pixel_scale=1.0)

            interp_defls = interpolation.InterpolationDeflections(array=interp_defls, coords=interp_coords,
                                                                  geometry=geometry, scheme=scheme)

            grid_to_interp = interp_defls.grid_to_interp_from_grid(
                grid=np.array([[0.501, 0.501], [-1.49, 0.49], [-0.499, 0.499],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                               [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

            assert (grid_to_interp[0] == np.array([8])).all()
            assert (grid_to_interp[1] == np.array([1])).all()
            assert (grid_to_interp[2] == np.array([4])).all()
            assert (grid_to_interp[3] == np.array([3])).all()
            assert (grid_to_interp[4] == np.array([4])).all()
            assert (grid_to_interp[5] == np.array([5])).all()
            assert (grid_to_interp[6] == np.array([6])).all()
            assert (grid_to_interp[7] == np.array([7])).all()
            assert (grid_to_interp[8] == np.array([8])).all()

    # class TestInterpolateValues:
    #
    #     class TestInterpolateTopLeft:
    #
    #         def test__1_coordinate__at_top_left_of_pixel__interpolates_to_average_of_4_values(self):
    #
    #             # |0|1|2|
    #             # |3|4|5|
    #             # |6|7|8|
    #
    #             # interp defls arn't ussed for bilinear interpolation weights
    #
    #             interp_defls = np.array([[ 1.0,  1.0], [ 1.0, 1.0], [ 0.0, 0.0],
    #                                      [ 1.0,  1.0], [ 1.0, 1.0], [ 0.0, 0.0],
    #                                      [ 0.0,  0.0], [ 0.0, 0.0], [ 0.0, 0.0]])
    #
    #             interp_coords = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
    #                                       [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
    #                                       [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])
    #
    #             geometry = interpolation.InterpolationGeometry(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0,
    #                                                            y_pixel_scale=1.0, x_pixel_scale=1.0)
    #             scheme = interpolation.InterpolationScheme(shape=(3, 3), datas_=interp_coords,
    #                                                        image_pixel_scale=1.0)
    #
    #             interp_defls = interpolation.InterpolationDeflections(array=interp_defls, grids=interp_coords,
    #                                                                   geometry=geometry, scheme=scheme)
    #
    #             interpolated = interp_defls.interpolate_values_from_grid(grid=np.array([[-0.499, -0.499]]))
    #
    #             assert interpolated[0,0] == 1.0
    #             assert interpolated[0,0] == 1.0
