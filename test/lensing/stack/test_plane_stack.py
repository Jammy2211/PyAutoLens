import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import exc
from autolens.data.array import grids, mask
from autolens.model.galaxy import galaxy as g
from autolens.lensing.util import plane_util
from autolens.lensing.stack import plane_stack as pl_stack
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.mock.mock_inversion import MockRegularization, MockPixelization
from test.mock.mock_imaging import MockBorders

@pytest.fixture(name="grid_stack_0")
def make_grid_stack_0():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=6.0)

    grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2, psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grid_stacks and defledctions straightforward

    grid_stack_0.regular[0] = np.array([1.0, 1.0])
    grid_stack_0.regular[1] = np.array([1.0, 0.0])
    grid_stack_0.sub[0] = np.array([1.0, 1.0])
    grid_stack_0.sub[1] = np.array([1.0, 0.0])
    grid_stack_0.sub[2] = np.array([1.0, 1.0])
    grid_stack_0.sub[3] = np.array([1.0, 0.0])
    grid_stack_0.sub[4] = np.array([-1.0, 2.0])
    grid_stack_0.sub[5] = np.array([-1.0, 4.0])
    grid_stack_0.sub[6] = np.array([1.0, 2.0])
    grid_stack_0.sub[7] = np.array([1.0, 4.0])
    grid_stack_0.blurring[0] = np.array([1.0, 0.0])
    grid_stack_0.blurring[1] = np.array([-6.0, -3.0])
    grid_stack_0.blurring[2] = np.array([-6.0, 3.0])
    grid_stack_0.blurring[3] = np.array([-6.0, 9.0])
    grid_stack_0.blurring[4] = np.array([0.0, -9.0])
    grid_stack_0.blurring[5] = np.array([0.0, 9.0])
    grid_stack_0.blurring[6] = np.array([6.0, -9.0])
    grid_stack_0.blurring[7] = np.array([6.0, -3.0])
    grid_stack_0.blurring[8] = np.array([6.0, 3.0])
    grid_stack_0.blurring[9] = np.array([6.0, 9.0])

    return grid_stack_0


@pytest.fixture(name="grid_stack_1")
def make_grid_stack_1():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=12.0)

    grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2, psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grid_stacks and defledctions straightforward

    grid_stack_0.regular[0] = np.array([2.0, 2.0])
    grid_stack_0.regular[1] = np.array([2.0, 2.0])
    grid_stack_0.sub[0] = np.array([2.0, 2.0])
    grid_stack_0.sub[1] = np.array([2.0, 0.0])
    grid_stack_0.sub[2] = np.array([2.0, 2.0])
    grid_stack_0.sub[3] = np.array([2.0, 0.0])
    grid_stack_0.sub[4] = np.array([-2.0, 4.0])
    grid_stack_0.sub[5] = np.array([-2.0, 8.0])
    grid_stack_0.sub[6] = np.array([2.0, 4.0])
    grid_stack_0.sub[7] = np.array([2.0, 8.0])
    grid_stack_0.blurring[0] = np.array([2.0, 0.0])
    grid_stack_0.blurring[1] = np.array([-12.0, -6.0])
    grid_stack_0.blurring[2] = np.array([-12.0, 6.0])
    grid_stack_0.blurring[3] = np.array([-12.0, 18.0])
    grid_stack_0.blurring[4] = np.array([0.0, -18.0])
    grid_stack_0.blurring[5] = np.array([0.0, 18.0])
    grid_stack_0.blurring[6] = np.array([12.0, -18.0])
    grid_stack_0.blurring[7] = np.array([12.0, -6.0])
    grid_stack_0.blurring[8] = np.array([12.0, 6.0])
    grid_stack_0.blurring[9] = np.array([12.0, 18.0])

    return grid_stack_0


@pytest.fixture(name="padded_grid_stack")
def make_padded_grid_stack():
    ma = mask.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(ma, 2, (3, 3))


@pytest.fixture(name='galaxy_non', scope='function')
def make_galaxy_non():
    return g.Galaxy()


@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))


@pytest.fixture(name="galaxy_mass")
def make_galaxy_mass():
    return g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='galaxy_mass_x2')
def make_galaxy_mass_x2():
    return g.Galaxy(sis_0=mp.SphericalIsothermal(einstein_radius=1.0),
                    sis_1=mp.SphericalIsothermal(einstein_radius=1.0))


class TestPlaneStack:

    class TestGridsStacksSetup:

        def test__grid_stack_0_setup_for_regular_sub_and_blurring__no_deflections(self, grid_stack_0,
                                                                                  galaxy_mass):
            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_0],
                                  compute_deflections=False)

            assert plane_stack.grid_stacks[0].regular == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0]]), 1e-3)
            assert plane_stack.grid_stacks[0].sub == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                                                                       [-1.0, 2.0], [-1.0, 4.0], [1.0, 2.0],
                                                                       [1.0, 4.0]]), 1e-3)
            assert plane_stack.grid_stacks[0].blurring == pytest.approx(
                np.array([[1.0, 0.0], [-6.0, -3.0], [-6.0, 3.0], [-6.0, 9.0],
                          [0.0, -9.0], [0.0, 9.0],
                          [6.0, -9.0], [6.0, -3.0], [6.0, 3.0], [6.0, 9.0]]), 1e-3)

            assert plane_stack.grid_stacks[1].regular == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0]]), 1e-3)
            assert plane_stack.grid_stacks[1].sub == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                                                                       [-1.0, 2.0], [-1.0, 4.0], [1.0, 2.0],
                                                                       [1.0, 4.0]]), 1e-3)
            assert plane_stack.grid_stacks[1].blurring == pytest.approx(
                np.array([[1.0, 0.0], [-6.0, -3.0], [-6.0, 3.0], [-6.0, 9.0],
                          [0.0, -9.0], [0.0, 9.0],
                          [6.0, -9.0], [6.0, -3.0], [6.0, 3.0], [6.0, 9.0]]), 1e-3)

            assert plane_stack.deflection_stacks == None

        def test__same_as_above_but_test_deflections(self, grid_stack_0, galaxy_mass):
            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_0],
                                  compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack_0.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack_0.blurring)

            assert plane_stack.deflection_stacks[0].regular == pytest.approx(np.array([[0.707, 0.707], [1.0, 0.0]]), 1e-3)
            assert (plane_stack.deflection_stacks[0].sub == sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[0].blurring == blurring_galaxy_deflections).all()

            assert plane_stack.deflection_stacks[1].regular == pytest.approx(np.array([[0.707, 0.707], [1.0, 0.0]]), 1e-3)
            assert (plane_stack.deflection_stacks[1].sub == sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[1].blurring == blurring_galaxy_deflections).all()

        def test__same_as_above__x2_galaxy_in_plane__or_galaxy_x2_sis__deflections_double(self, grid_stack_0,
                                                                                          galaxy_mass,
                                                                                          galaxy_mass_x2):
            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass_x2], grid_stacks=[grid_stack_0, grid_stack_0],
                                  compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(grid_stack_0.sub)
            blurring_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(grid_stack_0.blurring)

            assert plane_stack.deflection_stacks[0].regular == pytest.approx \
                (np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                       1e-3)
            assert (plane_stack.deflection_stacks[0].sub == sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[0].blurring == blurring_galaxy_deflections).all()

            assert plane_stack.deflection_stacks[1].regular == pytest.approx \
                (np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                       1e-3)
            assert (plane_stack.deflection_stacks[1].sub == sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[1].blurring == blurring_galaxy_deflections).all()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass, galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_0],
                                  compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack_0.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack_0.blurring)

            assert plane_stack.deflection_stacks[0].regular == pytest.approx \
                (np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                       1e-3)
            assert (plane_stack.deflection_stacks[0].sub == 2.0 * sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[0].blurring == 2.0 * blurring_galaxy_deflections).all()

            assert plane_stack.deflection_stacks[1].regular == pytest.approx \
                (np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                       1e-3)
            assert (plane_stack.deflection_stacks[1].sub == 2.0 * sub_galaxy_deflections).all()
            assert (plane_stack.deflection_stacks[1].blurring == 2.0 * blurring_galaxy_deflections).all()

    class TestProperties:

        def test__total_images(self, grid_stack_0):

            plane_stack = pl_stack.PlaneStack(galaxies=[g.Galaxy()], grid_stacks=[grid_stack_0])
            assert plane_stack.total_grid_stacks == 1

            plane_stack = pl_stack.PlaneStack(galaxies=[g.Galaxy()], grid_stacks=[grid_stack_0, grid_stack_0])
            assert plane_stack.total_grid_stacks == 2

            plane_stack = pl_stack.PlaneStack(galaxies=[g.Galaxy()], grid_stacks=[grid_stack_0, grid_stack_0, grid_stack_0])
            assert plane_stack.total_grid_stacks == 3

        def test__padded_grid_in__tracer_has_padded_grid_property(self, grid_stack_0, padded_grid_stack, galaxy_light):

            plane_stack = pl_stack.PlaneStack(grid_stacks=[grid_stack_0], galaxies=[galaxy_light])
            assert plane_stack.has_padded_grid_stack == False

            plane_stack = pl_stack.PlaneStack(grid_stacks=[padded_grid_stack], galaxies=[galaxy_light])
            assert plane_stack.has_padded_grid_stack == True

            plane_stack = pl_stack.PlaneStack(grid_stacks=[grid_stack_0, padded_grid_stack], galaxies=[galaxy_light])
            assert plane_stack.has_padded_grid_stack == True

    class TestImages:

        def test__images_from_plane__same_as_light_profile_images(self, grid_stack_0, galaxy_light):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.sub[5] = np.array([2.0, 2.0])

            lp = galaxy_light.light_profiles[0]

            lp_sub_image = lp.intensities_from_grid(grid_stack_0.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_light], grid_stacks=[grid_stack_0])

            assert (plane_stack.image_plane_images_1d[0][0] == lp_image_pixel_0).all()
            assert (plane_stack.image_plane_images_1d[0][1] == lp_image_pixel_1).all()
            assert (plane_stack.image_plane_images[0] ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.image_plane_images_1d[0])).all()

        def test__image_from_plane__same_as_galaxy_images(self, grid_stack_0, galaxy_light):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.sub[5] = np.array([2.0, 2.0])

            galaxy_image = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.sub, galaxies=[galaxy_light])

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_light], grid_stacks=[grid_stack_0, grid_stack_0])

            assert (plane_stack.image_plane_images_1d[0] == galaxy_image).all()
            assert (plane_stack.image_plane_images[0] ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.image_plane_images_1d[0])).all()

            assert (plane_stack.image_plane_images_1d[1] == galaxy_image).all()
            assert (plane_stack.image_plane_images[1] ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.image_plane_images_1d[1])).all()

        def test__image_plane_image_of_galaxies__use_multiple_grids__get_multiple_images(self, grid_stack_0,
                                                                                         grid_stack_1):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            plane_stack = pl_stack.PlaneStack(galaxies=[g0, g1], grid_stacks=[grid_stack_0, grid_stack_1])

            g0_image_grid_0 = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.sub, galaxies=[g0])
            g1_image_grid_0 = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.sub, galaxies=[g1])

            assert plane_stack.image_plane_images_1d[0] == pytest.approx(g0_image_grid_0 + g1_image_grid_0, 1.0e-4)
            assert (plane_stack.image_plane_images[0] ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.image_plane_images_1d[0])).all()

            assert (plane_stack.image_plane_images_1d_of_galaxies[0][0] == g0_image_grid_0).all()
            assert (plane_stack.image_plane_images_1d_of_galaxies[0][1] == g1_image_grid_0).all()

            g0_image_grid_1 = plane_util.intensities_of_galaxies_from_grid(grid_stack_1.sub, galaxies=[g0])
            g1_image_grid_1 = plane_util.intensities_of_galaxies_from_grid(grid_stack_1.sub, galaxies=[g1])

            assert plane_stack.image_plane_images_1d[1] == pytest.approx(g0_image_grid_1 + g1_image_grid_1, 1.0e-4)
            assert (plane_stack.image_plane_images[1] ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.image_plane_images_1d[1])).all()

            assert (plane_stack.image_plane_images_1d_of_galaxies[1][0] == g0_image_grid_1).all()
            assert (plane_stack.image_plane_images_1d_of_galaxies[1][1] == g1_image_grid_1).all()

        def test__padded_grid_stack_in__image_plane_image_is_padded(self, padded_grid_stack, galaxy_light):

            lp = galaxy_light.light_profiles[0]

            lp_sub_image = lp.intensities_from_grid(padded_grid_stack.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4
            lp_image_pixel_2 = (lp_sub_image[8] + lp_sub_image[9] + lp_sub_image[10] + lp_sub_image[11]) / 4
            lp_image_pixel_3 = (lp_sub_image[12] + lp_sub_image[13] + lp_sub_image[14] + lp_sub_image[15]) / 4
            lp_image_pixel_4 = (lp_sub_image[16] + lp_sub_image[17] + lp_sub_image[18] + lp_sub_image[19]) / 4
            lp_image_pixel_5 = (lp_sub_image[20] + lp_sub_image[21] + lp_sub_image[22] + lp_sub_image[23]) / 4
            lp_image_pixel_6 = (lp_sub_image[24] + lp_sub_image[25] + lp_sub_image[26] + lp_sub_image[27]) / 4
            lp_image_pixel_7 = (lp_sub_image[28] + lp_sub_image[29] + lp_sub_image[30] + lp_sub_image[31]) / 4
            lp_image_pixel_8 = (lp_sub_image[32] + lp_sub_image[33] + lp_sub_image[34] + lp_sub_image[35]) / 4
            lp_image_pixel_9 = (lp_sub_image[36] + lp_sub_image[37] + lp_sub_image[38] + lp_sub_image[39]) / 4
            lp_image_pixel_10 = (lp_sub_image[40] + lp_sub_image[41] + lp_sub_image[42] + lp_sub_image[43]) / 4
            lp_image_pixel_11 = (lp_sub_image[44] + lp_sub_image[45] + lp_sub_image[46] + lp_sub_image[47]) / 4

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_light], grid_stacks=[padded_grid_stack, padded_grid_stack])

            assert plane_stack.image_plane_images_for_simulation[0].shape == (3, 4)
            assert (plane_stack.image_plane_images_for_simulation[0][0, 0] == lp_image_pixel_0).all()
            assert (plane_stack.image_plane_images_for_simulation[0][0, 1] == lp_image_pixel_1).all()
            assert (plane_stack.image_plane_images_for_simulation[0][0, 2] == lp_image_pixel_2).all()
            assert (plane_stack.image_plane_images_for_simulation[0][0, 3] == lp_image_pixel_3).all()
            assert (plane_stack.image_plane_images_for_simulation[0][1, 0] == lp_image_pixel_4).all()
            assert (plane_stack.image_plane_images_for_simulation[0][1, 1] == lp_image_pixel_5).all()
            assert (plane_stack.image_plane_images_for_simulation[0][1, 2] == lp_image_pixel_6).all()
            assert (plane_stack.image_plane_images_for_simulation[0][1, 3] == lp_image_pixel_7).all()
            assert (plane_stack.image_plane_images_for_simulation[0][2, 0] == lp_image_pixel_8).all()
            assert (plane_stack.image_plane_images_for_simulation[0][2, 1] == lp_image_pixel_9).all()
            assert (plane_stack.image_plane_images_for_simulation[0][2, 2] == lp_image_pixel_10).all()
            assert (plane_stack.image_plane_images_for_simulation[0][2, 3] == lp_image_pixel_11).all()

            assert plane_stack.image_plane_images_for_simulation[1].shape == (3, 4)
            assert (plane_stack.image_plane_images_for_simulation[1][0, 0] == lp_image_pixel_0).all()
            assert (plane_stack.image_plane_images_for_simulation[1][0, 1] == lp_image_pixel_1).all()
            assert (plane_stack.image_plane_images_for_simulation[1][0, 2] == lp_image_pixel_2).all()
            assert (plane_stack.image_plane_images_for_simulation[1][0, 3] == lp_image_pixel_3).all()
            assert (plane_stack.image_plane_images_for_simulation[1][1, 0] == lp_image_pixel_4).all()
            assert (plane_stack.image_plane_images_for_simulation[1][1, 1] == lp_image_pixel_5).all()
            assert (plane_stack.image_plane_images_for_simulation[1][1, 2] == lp_image_pixel_6).all()
            assert (plane_stack.image_plane_images_for_simulation[1][1, 3] == lp_image_pixel_7).all()
            assert (plane_stack.image_plane_images_for_simulation[1][2, 0] == lp_image_pixel_8).all()
            assert (plane_stack.image_plane_images_for_simulation[1][2, 1] == lp_image_pixel_9).all()
            assert (plane_stack.image_plane_images_for_simulation[1][2, 2] == lp_image_pixel_10).all()
            assert (plane_stack.image_plane_images_for_simulation[1][2, 3] == lp_image_pixel_11).all()

    class TestBlurringImages:

        def test__images_from_plane__same_as_their_light_profile(self, grid_stack_0, grid_stack_1,
                                                                 galaxy_light):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.blurring[1] = np.array([2.0, 2.0])

            lp = galaxy_light.light_profiles[0]

            lp_blurring_image = lp.intensities_from_grid(grid_stack_0.blurring)
            lp_blurring_image_1 = lp.intensities_from_grid(grid_stack_1.blurring)

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_light], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.image_plane_blurring_images_1d[0] == lp_blurring_image).all()
            assert (plane_stack.image_plane_blurring_images_1d[1] == lp_blurring_image_1).all()

        def test__same_as_above_but_for_galaxies(self, grid_stack_0, grid_stack_1, galaxy_light):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.blurring[1] = np.array([2.0, 2.0])

            galaxy_image = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.blurring, galaxies=[galaxy_light])
            galaxy_image_1 = plane_util.intensities_of_galaxies_from_grid(grid_stack_1.blurring, galaxies=[galaxy_light])

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_light], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.image_plane_blurring_images_1d[0] == galaxy_image).all()
            assert (plane_stack.image_plane_blurring_images_1d[1] == galaxy_image_1).all()

        def test__same_as_above__multiple_galaxies(self, grid_stack_0, grid_stack_1):

            # Overwrite one value so intensity in each pixel is different
            grid_stack_0.blurring[1] = np.array([2.0, 2.0])
            grid_stack_1.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image_grid_0 = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.blurring, galaxies=[g0])
            g1_image_grid_0 = plane_util.intensities_of_galaxies_from_grid(grid_stack_0.blurring, galaxies=[g1])

            plane_stack = pl_stack.PlaneStack(galaxies=[g0, g1], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.image_plane_blurring_images_1d[0] == g0_image_grid_0 + g1_image_grid_0).all()

            g0_image_grid_1 = plane_util.intensities_of_galaxies_from_grid(grid_stack_1.blurring, galaxies=[g0])
            g1_image_grid_1 = plane_util.intensities_of_galaxies_from_grid(grid_stack_1.blurring, galaxies=[g1])

            plane_stack = pl_stack.PlaneStack(galaxies=[g0, g1], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.image_plane_blurring_images_1d[1] == g0_image_grid_1 + g1_image_grid_1).all()

    class TestSurfaceDensity:

        def test__computes_x1_surface_density__uses_primary_grid_stack_0(self, grid_stack_0, grid_stack_1,
                                                                         galaxy_mass):

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.surface_density_from_grid(grid_stack_0.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_image_pixel_0 = (mp_sub_image[0] + mp_sub_image[1] + mp_sub_image[2] + mp_sub_image[3]) / 4
            mp_image_pixel_1 = (mp_sub_image[4] + mp_sub_image[5] + mp_sub_image[6] + mp_sub_image[7]) / 4

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.surface_density[1, 1] == mp_image_pixel_0).all()
            assert (plane_stack.surface_density[1, 2] == mp_image_pixel_1).all()

    class TestPotential:

        def test__computes_x1_potential__uses_primary_grid_stack_0(self, grid_stack_0, grid_stack_1,
                                                                   galaxy_mass):

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.potential_from_grid(grid_stack_0.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_image_pixel_0 = (mp_sub_image[0] + mp_sub_image[1] + mp_sub_image[2] + mp_sub_image[3]) / 4
            mp_image_pixel_1 = (mp_sub_image[4] + mp_sub_image[5] + mp_sub_image[6] + mp_sub_image[7]) / 4

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.potential[1, 1] == mp_image_pixel_0).all()
            assert (plane_stack.potential[1, 2] == mp_image_pixel_1).all()

    class TestDeflections:

        def test__computes_x1_deflections__uses_primary_grid_stack_0(self, grid_stack_0, grid_stack_1,
                                                                     galaxy_mass):

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.deflections_from_grid(grid_stack_0.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_image_pixel_0x = (mp_sub_image[0 ,0] + mp_sub_image[1, 0] + mp_sub_image[2, 0] + mp_sub_image[3, 0]) / 4
            mp_image_pixel_1x = (mp_sub_image[4, 0] + mp_sub_image[5, 0] + mp_sub_image[6, 0] + mp_sub_image[7, 0]) / 4
            mp_image_pixel_0y = (mp_sub_image[0, 1] + mp_sub_image[1, 1] + mp_sub_image[2, 1] + mp_sub_image[3, 1]) / 4
            mp_image_pixel_1y = (mp_sub_image[4, 1] + mp_sub_image[5, 1] + mp_sub_image[6, 1] + mp_sub_image[7, 1]) / 4

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_mass], grid_stacks=[grid_stack_0, grid_stack_1])

            assert (plane_stack.deflections_1d[0, 0] == mp_image_pixel_0x).all()
            assert (plane_stack.deflections_1d[0, 1] == mp_image_pixel_0y).all()
            assert (plane_stack.deflections_1d[1, 0] == mp_image_pixel_1x).all()
            assert (plane_stack.deflections_1d[1, 1] == mp_image_pixel_1y).all()
            assert (plane_stack.deflections_y ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.deflections_1d[:, 0])).all()
            assert (plane_stack.deflections_x ==
                    grid_stack_0.regular.scaled_array_from_array_1d(plane_stack.deflections_1d[:, 1])).all()

    class TestMapper:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, grid_stack_0):
            galaxy_no_pix = g.Galaxy()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_no_pix], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            assert plane_stack.mapper is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, grid_stack_0):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_pix], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            assert plane_stack.mapper == 1

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, grid_stack_0):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_no_pix, galaxy_pix], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            assert plane_stack.mapper == 1

        def test__plane_has_no_border__still_returns_mapper(self, grid_stack_0):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_no_pix, galaxy_pix], grid_stacks=[grid_stack_0])

            assert plane_stack.mapper == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, grid_stack_0):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_pix_0, galaxy_pix_1], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            with pytest.raises(exc.PixelizationException):
                plane_stack.mapper

    class TestRegularization:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, grid_stack_0):
            galaxy_no_pix = g.Galaxy()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_no_pix], grid_stacks=[grid_stack_0], borders=[MockBorders()])

            assert plane_stack.regularization is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, grid_stack_0):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_pix], grid_stacks=[grid_stack_0], borders=[MockBorders()])

            assert plane_stack.regularization.value == 0

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, grid_stack_0):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_no_pix, galaxy_pix], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            assert plane_stack.regularization.value == 0

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, grid_stack_0):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy_pix_0, galaxy_pix_1], grid_stacks=[grid_stack_0],
                                  borders=[MockBorders()])

            with pytest.raises(exc.PixelizationException):
                plane_stack.regularization

    class TestPlaneImages:

        def test__x2_data_stacks__x2_plane_images(self, grid_stack_0, grid_stack_1):
            grid_stack_0.regular[1] = np.array([2.0, 2.0])

            galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

            plane_stack = pl_stack.PlaneStack(galaxies=[galaxy], grid_stacks=[grid_stack_0, grid_stack_1],
                                  compute_deflections=False)

            plane_image_from_func = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 4),
                                                                                 grid=grid_stack_0.regular,
                                                                          galaxies=[galaxy])

            plane_image_from_func_1 = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 4),
                                                                                   grid=grid_stack_1.regular,
                                                                            galaxies=[galaxy])

            assert (plane_image_from_func == plane_stack.plane_images[0]).all()
            assert (plane_image_from_func_1 == plane_stack.plane_images[1]).all()

        def test__ensure_index_of_plane_image_has_negative_arcseconds_at_start(self, grid_stack_0):
            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The origin -1.6, -1.6 of the model_galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane datas_.

            msk = mask.Mask(array=np.full((5, 5), False), pixel_scale=1.0)

            grid_stack_0.regular = grids.RegularGrid(np.array([[-2.0, -2.0], [2.0, 2.0]]), mask=msk)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0))
            plane_stack = pl_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0])

            assert plane_stack.plane_images[0].shape == (5, 5)
            assert np.unravel_index(plane_stack.plane_images[0].argmax(), plane_stack.plane_images[0].shape) == (0, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0))
            plane_stack = pl_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0])
            assert np.unravel_index(plane_stack.plane_images[0].argmax(), plane_stack.plane_images[0].shape) == (0, 4)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, -1.6), intensity=1.0))
            plane_stack = pl_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0])
            assert np.unravel_index(plane_stack.plane_images[0].argmax(), plane_stack.plane_images[0].shape) == (4, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0))
            plane_stack = pl_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0])
            assert np.unravel_index(plane_stack.plane_images[0].argmax(), plane_stack.plane_images[0].shape) == (4, 4)
