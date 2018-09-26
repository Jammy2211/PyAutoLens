import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import exc
from autolens.imaging import imaging_util
from autolens.imaging import mask
from autolens.inversion import pixelizations
from autolens.inversion import regularization
from autolens.lensing import galaxy as g
from autolens.lensing import plane as pl
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from test.mock.mock_lensing import MockRegularization, MockPixelization, MockBorders

@pytest.fixture(name="imaging_grids")
def make_imaging_grids():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=6.0)

    imaging_grids = mask.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                  psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grids and defledctions straightforward

    imaging_grids.image[0] = np.array([1.0, 1.0])
    imaging_grids.image[1] = np.array([1.0, 0.0])
    imaging_grids.sub[0] = np.array([1.0, 1.0])
    imaging_grids.sub[1] = np.array([1.0, 0.0])
    imaging_grids.sub[2] = np.array([1.0, 1.0])
    imaging_grids.sub[3] = np.array([1.0, 0.0])
    imaging_grids.blurring[0] = np.array([1.0, 0.0])

    return imaging_grids


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


class TestIntensitiesFromGrid:

    def test__no_galaxies__intensities_returned_as_0s(self, imaging_grids, galaxy_non):

        imaging_grids.image = np.array([[1.0, 1.0],
                                        [2.0, 2.0],
                                        [3.0, 3.0]])

        intensities = pl.intensities_from_grid(grid=imaging_grids.image,
                                                        galaxies=[galaxy_non])

        assert (intensities[0] == np.array([0.0, 0.0])).all()
        assert (intensities[1] == np.array([0.0, 0.0])).all()
        assert (intensities[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_light__intensities_returned_as_correct_values(self, imaging_grids, galaxy_light):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_intensities = galaxy_light.intensities_from_grid(imaging_grids.image)

        tracer_intensities = pl.intensities_from_grid(grid=imaging_grids.image,
                                                               galaxies=[galaxy_light])

        assert (galaxy_intensities == tracer_intensities).all()

    def test__galaxy_light_x2__intensities_double_from_above(self, imaging_grids, galaxy_light):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_intensities = galaxy_light.intensities_from_grid(imaging_grids.image)

        tracer_intensities = pl.intensities_from_grid(grid=imaging_grids.image,
                                                               galaxies=[galaxy_light, galaxy_light])

        assert (2.0 * galaxy_intensities == tracer_intensities).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, imaging_grids, galaxy_light):
        galaxy_image = galaxy_light.intensities_from_grid(imaging_grids.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        tracer_intensities = pl.intensities_from_grid(grid=imaging_grids.sub, galaxies=[galaxy_light])

        assert tracer_intensities[0] == galaxy_image


class TestSurfaceDensityFromGrid:

    def test__no_galaxies__surface_density_returned_as_0s(self, imaging_grids, galaxy_non):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [2.0, 2.0],
                                        [3.0, 3.0]])

        surface_density = pl.surface_density_from_grid(grid=imaging_grids.image, galaxies=[galaxy_non])

        assert (surface_density[0] == np.array([0.0, 0.0])).all()
        assert (surface_density[1] == np.array([0.0, 0.0])).all()
        assert (surface_density[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_mass__surface_density_returned_as_correct_values(self, imaging_grids, galaxy_mass):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_surface_density = galaxy_mass.surface_density_from_grid(imaging_grids.image)

        tracer_surface_density = pl.surface_density_from_grid(grid=imaging_grids.image,
                                                                       galaxies=[galaxy_mass])

        assert (galaxy_surface_density == tracer_surface_density).all()

    def test__galaxy_mass_x2__surface_density_double_from_above(self, imaging_grids, galaxy_mass):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_surface_density = galaxy_mass.surface_density_from_grid(imaging_grids.image)

        tracer_surface_density = pl.surface_density_from_grid(grid=imaging_grids.image,
                                                                       galaxies=[galaxy_mass, galaxy_mass])

        assert (2.0 * galaxy_surface_density == tracer_surface_density).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, imaging_grids, galaxy_mass):
        galaxy_image = galaxy_mass.surface_density_from_grid(imaging_grids.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        tracer_surface_density = pl.surface_density_from_grid(grid=imaging_grids.sub,
                                                                       galaxies=[galaxy_mass])

        assert tracer_surface_density[0] == galaxy_image


class TestPotentialFromGrid:

    def test__no_galaxies__potential_returned_as_0s(self, imaging_grids, galaxy_non):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [2.0, 2.0],
                                        [3.0, 3.0]])

        potential = pl.potential_from_grid(grid=imaging_grids.image, galaxies=[galaxy_non])

        assert (potential[0] == np.array([0.0, 0.0])).all()
        assert (potential[1] == np.array([0.0, 0.0])).all()
        assert (potential[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_mass__potential_returned_as_correct_values(self, imaging_grids, galaxy_mass):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_potential = galaxy_mass.potential_from_grid(imaging_grids.image)

        tracer_potential = pl.potential_from_grid(grid=imaging_grids.image, galaxies=[galaxy_mass])

        assert (galaxy_potential == tracer_potential).all()

    def test__galaxy_mass_x2__potential_double_from_above(self, imaging_grids, galaxy_mass):
        imaging_grids.image = np.array([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [-1.0, 0.0]])

        galaxy_potential = galaxy_mass.potential_from_grid(imaging_grids.image)

        tracer_potential = pl.potential_from_grid(grid=imaging_grids.image,
                                                           galaxies=[galaxy_mass, galaxy_mass])

        assert (2.0 * galaxy_potential == tracer_potential).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, imaging_grids, galaxy_mass):
        galaxy_image = galaxy_mass.potential_from_grid(imaging_grids.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        tracer_potential = pl.potential_from_grid(grid=imaging_grids.sub, galaxies=[galaxy_mass])

        assert tracer_potential[0] == galaxy_image


class TestDeflectionsFromGrid:

    def test__all_coordinates(self, imaging_grids, galaxy_mass):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass])

        assert deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        #    assert deflections.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    def test__2_identical_lens_galaxies__deflection_angles_double(self, imaging_grids, galaxy_mass):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass, galaxy_mass])

        assert deflections.image[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
        #    assert deflections.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

    def test__1_lens_with_2_identical_mass_profiles__deflection_angles_double(self, imaging_grids, galaxy_mass_x2):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass_x2])

        assert deflections.image[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
        assert deflections.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)


class TestSetupTracedGrid:

    def test__simple_sis_model__deflection_angles(self, imaging_grids, galaxy_mass):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass])

        grid_traced = pl.traced_collection_for_deflections(imaging_grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)

    def test_two_identical_lenses__deflection_angles_double(self, imaging_grids, galaxy_mass):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass, galaxy_mass])

        grid_traced = pl.traced_collection_for_deflections(imaging_grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3)

    def test_one_lens_with_double_identical_mass_profiles__deflection_angles_double(self, imaging_grids,
                                                                                    galaxy_mass_x2):
        deflections = pl.deflections_from_grid_collection(imaging_grids, [galaxy_mass_x2])

        grid_traced = pl.traced_collection_for_deflections(imaging_grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3)


class TestUniformGridFromLensedGrid:

    def test__3x3_grid__extracts_max_min_coordinates__creates_regular_grid_including_half_pixel_offset_from_edge(self):
        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = pl.uniform_grid_from_lensed_grid(grid, shape=(3, 3))

        assert (source_plane_grid == np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                               [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])).all()

    def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self):
        grid = np.array([[-1.5, -1.5], [1.5, 1.5], [0.1, -0.1], [-1.0, 0.6], [1.4, -1.3], [1.5, 1.5]])

        source_plane_grid = pl.uniform_grid_from_lensed_grid(grid, shape=(3, 3))

        assert (source_plane_grid == np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                               [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])).all()

    def test__2x3_grid__shape_change_correct_and_coordinates_shift(self):
        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = pl.uniform_grid_from_lensed_grid(grid, shape=(2, 3))

        assert (source_plane_grid == np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                               [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]])).all()

    def test__3x2_grid__shape_change_correct_and_coordinates_shift(self):
        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = pl.uniform_grid_from_lensed_grid(grid, shape=(3, 2))

        assert (source_plane_grid == np.array([[-1.0, -0.75], [-1.0, 0.75],
                                               [0.0, -0.75], [0.0, 0.75],
                                               [1.0, -0.75], [1.0, 0.75]])).all()


class TestPlane(object):

    class TestGridsSetup:

        def test__imaging_grids_setup_for_image_sub_and_blurring__no_deflections(self, imaging_grids, galaxy_mass):
            
            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids, compute_deflections=False)

            assert plane.grids.image == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0]]), 1e-3)
            assert plane.grids.sub == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                                                              [-1.0, 2.0], [-1.0, 4.0], [1.0, 2.0], [1.0, 4.0]]), 1e-3)
            assert plane.grids.blurring == pytest.approx(np.array([[1.0, 0.0], [-6.0, -3.0], [-6.0, 3.0], [-6.0, 9.0],
                                                                   [0.0, -9.0], [0.0, 9.0],
                                                                   [6.0, -9.0], [6.0, -3.0], [6.0, 3.0], [6.0, 9.0]]),
                                                         1e-3)

            assert plane.deflections == None

        def test__same_as_above_but_test_deflections(self, imaging_grids, galaxy_mass):
            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids,
                                      compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(imaging_grids.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(imaging_grids.blurring)

            assert plane.deflections.image == pytest.approx(np.array([[0.707, 0.707], [1.0, 0.0]]), 1e-3)
            assert (plane.deflections.sub == sub_galaxy_deflections).all()
            assert (plane.deflections.blurring == blurring_galaxy_deflections).all()

        def test__same_as_above__x2_galaxy_in_plane__or_galaxy_x2_sis__deflections_double(self, imaging_grids,
                                                                                          galaxy_mass,
                                                                                          galaxy_mass_x2):
            plane = pl.Plane(galaxies=[galaxy_mass_x2], grids=imaging_grids, compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(imaging_grids.sub)
            blurring_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(imaging_grids.blurring)

            assert plane.deflections.image == pytest.approx(np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]), 1e-3)
            assert (plane.deflections.sub == sub_galaxy_deflections).all()
            assert (plane.deflections.blurring == blurring_galaxy_deflections).all()

            plane = pl.Plane(galaxies=[galaxy_mass, galaxy_mass], grids=imaging_grids,
                                      compute_deflections=True)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(imaging_grids.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(imaging_grids.blurring)

            assert plane.deflections.image == pytest.approx(np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]), 1e-3)
            assert (plane.deflections.sub == 2.0 * sub_galaxy_deflections).all()
            assert (plane.deflections.blurring == 2.0 * blurring_galaxy_deflections).all()

    class TestRedshift:

        def test__galaxy_redshifts_gives_list_of_redshifts(self, imaging_grids):

            g0 = g.Galaxy(redshift=1.0)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=1.0)

            plane = pl.Plane(grids=imaging_grids, galaxies=[g0, g1, g2])

            assert plane.galaxy_redshifts == [1.0, 1.0, 1.0]

        def test__galaxy_has_no_redshift__cosmology_input__raises_exception(self, imaging_grids):

            g0 = g.Galaxy()
            g1 = g.Galaxy(redshift=1.0)

            with pytest.raises(exc.RayTracingException):
                pl.Plane(grids=imaging_grids, galaxies=[g0, g1], cosmology=cosmo.LambdaCDM)


        def test__galaxies_entered_all_have_no_redshifts__no_exception_raised(self, imaging_grids):

            g0 = g.Galaxy()
            g1 = g.Galaxy()

            pl.Plane(grids=imaging_grids, galaxies=[g0, g1])

        def test__galaxies_entered_all_have_same_redshifts__no_exception_raised(self, imaging_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=0.1)

            pl.Plane(grids=imaging_grids, galaxies=[g0, g1])

        def test__1_galaxy_has_redshift_other_does_not__exception_is_raised(self, imaging_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy()

            with pytest.raises(exc.RayTracingException):
                pl.Plane(grids=imaging_grids, galaxies=[g0, g1])

        def test__galaxies_have_different_redshifts__exception_is_raised(self, imaging_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            with pytest.raises(exc.RayTracingException):
                pl.Plane(grids=imaging_grids, galaxies=[g0, g1])

        def test__galaxy_has_redshift__returns_redshift(self, imaging_grids):

            g0 = g.Galaxy(redshift=0.1)

            plane = pl.Plane(grids=imaging_grids, galaxies=[g0])

            assert plane.redshift == 0.1

        def test__galaxy_has_no_redshift__returns_none(self, imaging_grids):

            g0 = g.Galaxy()

            plane = pl.Plane(grids=imaging_grids, galaxies=[g0])

            assert plane.redshift == None

    class TestCosmology:

        def test__arcsec_to_kpc_coversion_and_anguar_diameter_distance_to_earth(self, imaging_grids):
            
            g0 = g.Galaxy(redshift=0.1)            
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids, cosmology=cosmo.Planck15)
            assert plane.arcsec_per_kpc_proper == pytest.approx(0.525060, 1e-5)
            assert plane.kpc_per_arcsec_proper == pytest.approx(1.904544, 1e-5)
            assert plane.angular_diameter_distance_to_earth == pytest.approx(392840, 1e-5)

            g0 = g.Galaxy(redshift=1.0)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids, cosmology=cosmo.Planck15)
            assert plane.arcsec_per_kpc_proper == pytest.approx(0.1214785, 1e-5)
            assert plane.kpc_per_arcsec_proper == pytest.approx(8.231907, 1e-5)
            assert plane.angular_diameter_distance_to_earth == pytest.approx(1697952, 1e-5)

        def test__cosmology_is_none__arguments_return_none(self, imaging_grids):

            g0 = g.Galaxy(redshift=0.1)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.arcsec_per_kpc_proper == None
            assert plane.kpc_per_arcsec_proper == None
            assert plane.angular_diameter_distance_to_earth == None

            g0 = g.Galaxy(redshift=1.0)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.arcsec_per_kpc_proper == None
            assert plane.kpc_per_arcsec_proper == None
            assert plane.angular_diameter_distance_to_earth == None

    class TestBooleans:

        def test__no_galaxies__raises_exception(self):

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[], grids=None)

        def test__has_light_profile(self, imaging_grids):

            plane = pl.Plane(galaxies=[g.Galaxy()], grids=imaging_grids)
            assert plane.has_light_profile == False

            plane = pl.Plane(galaxies=[g.Galaxy(light_profile=lp.LightProfile())], grids=imaging_grids)
            assert plane.has_light_profile == True

            plane = pl.Plane(galaxies=[g.Galaxy(light_profile=lp.LightProfile()), g.Galaxy()], grids=imaging_grids)
            assert plane.has_light_profile == True
            
        def test__has_pixelization(self, imaging_grids):

            plane = pl.Plane(galaxies=[g.Galaxy()], grids=imaging_grids)
            assert plane.has_pixelization == False

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.Plane(galaxies=[galaxy_pix], grids=imaging_grids)
            assert plane.has_pixelization == True

            plane = pl.Plane(galaxies=[galaxy_pix, g.Galaxy()], grids=imaging_grids)
            assert plane.has_pixelization == True
            
        def test__has_regularization(self, imaging_grids):

            plane = pl.Plane(galaxies=[g.Galaxy()], grids=imaging_grids)
            assert plane.has_regularization == False

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.Plane(galaxies=[galaxy_pix], grids=imaging_grids)
            assert plane.has_regularization == True

            plane = pl.Plane(galaxies=[galaxy_pix, g.Galaxy()], grids=imaging_grids)
            assert plane.has_regularization == True

        def test__has_hyper_galaxy(self, imaging_grids):

            plane = pl.Plane(galaxies=[g.Galaxy()], grids=imaging_grids)
            assert plane.has_hyper_galaxy == False

            plane = pl.Plane(galaxies=[g.Galaxy(hyper_galaxy=g.HyperGalaxy())], grids=imaging_grids)
            assert plane.has_hyper_galaxy == True

            plane = pl.Plane(galaxies=[g.Galaxy(hyper_galaxy=g.HyperGalaxy()), g.Galaxy()], grids=imaging_grids)
            assert plane.has_hyper_galaxy == True

        def test__extract_hyper_galaxies(self, imaging_grids):

            plane = pl.Plane(galaxies=[g.Galaxy()], grids=imaging_grids)
            assert plane.hyper_galaxies == [None]

            hyper_galaxy = g.HyperGalaxy()
            plane = pl.Plane(galaxies=[g.Galaxy(hyper_galaxy=hyper_galaxy)], grids=imaging_grids)
            assert plane.hyper_galaxies == [hyper_galaxy]

            plane = pl.Plane(galaxies=[g.Galaxy(), g.Galaxy(hyper_galaxy=hyper_galaxy), g.Galaxy()],
                             grids=imaging_grids)
            assert plane.hyper_galaxies == [None, hyper_galaxy, None]

    class TestImages:

        def test__image_from_plane__same_as_its_light_profile_image(self, imaging_grids, galaxy_light):
            lp = galaxy_light.light_profiles[0]

            lp_sub_image = lp.intensities_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_image[0] == lp_image_pixel_0).all()
            assert (plane._image_plane_image[1] == lp_image_pixel_1).all()
            assert (plane._image_plane_images_of_galaxies[0][0] == lp_image_pixel_0).all()
            assert (plane._image_plane_images_of_galaxies[0][1] == lp_image_pixel_1).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_light):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            lp = galaxy_light.light_profiles[0]

            lp_sub_image = lp.intensities_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_image[0] == lp_image_pixel_0).all()
            assert (plane._image_plane_image[1] == lp_image_pixel_1).all()
            assert (plane._image_plane_images_of_galaxies[0][0] == lp_image_pixel_0).all()
            assert (plane._image_plane_images_of_galaxies[0][1] == lp_image_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_sub_image = lp0.intensities_from_grid(imaging_grids.sub)
            lp1_sub_image = lp1.intensities_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            lp0_image_pixel_0 = (lp0_sub_image[0] + lp0_sub_image[1] + lp0_sub_image[2] + lp0_sub_image[3]) / 4
            lp0_image_pixel_1 = (lp0_sub_image[4] + lp0_sub_image[5] + lp0_sub_image[6] + lp0_sub_image[7]) / 4
            lp1_image_pixel_0 = (lp1_sub_image[0] + lp1_sub_image[1] + lp1_sub_image[2] + lp1_sub_image[3]) / 4
            lp1_image_pixel_1 = (lp1_sub_image[4] + lp1_sub_image[5] + lp1_sub_image[6] + lp1_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._image_plane_image[0] == lp0_image_pixel_0 + lp1_image_pixel_0).all()
            assert (plane._image_plane_image[1] == lp0_image_pixel_1 + lp1_image_pixel_1).all()
            assert (plane._image_plane_images_of_galaxies[0][0] == lp0_image_pixel_0).all()
            assert (plane._image_plane_images_of_galaxies[1][0] == lp1_image_pixel_0).all()
            assert (plane._image_plane_images_of_galaxies[0][1] == lp0_image_pixel_1).all()
            assert (plane._image_plane_images_of_galaxies[1][1] == lp1_image_pixel_1).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, imaging_grids, galaxy_light):
            
            galaxy_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[galaxy_light])

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_image == galaxy_image).all()
            assert (plane._image_plane_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_light):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            galaxy_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[galaxy_light])

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_image == galaxy_image).all()
            assert (plane._image_plane_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[g1])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._image_plane_image == g0_image + g1_image).all()
            assert (plane._image_plane_images_of_galaxies[0] == g0_image).all()
            assert (plane._image_plane_images_of_galaxies[1] == g1_image).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_intensities_for_each(self, imaging_grids):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            g0_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[g1])
            g2_image = pl.intensities_from_grid(imaging_grids.sub, galaxies=[g2])

            plane = pl.Plane(galaxies=[g0, g1, g2], grids=imaging_grids)

            assert (plane._image_plane_image == g0_image + g1_image + g2_image).all()
            assert (plane._image_plane_images_of_galaxies[0] == g0_image).all()
            assert (plane._image_plane_images_of_galaxies[1] == g1_image).all()
            assert (plane._image_plane_images_of_galaxies[2] == g2_image).all()

    class TestBlurringImage:

        def test__image_from_plane__same_as_its_light_profile_image(self, imaging_grids, galaxy_light):
            lp = galaxy_light.light_profiles[0]

            lp_blurring_image = lp.intensities_from_grid(imaging_grids.blurring)

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == lp_blurring_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == lp_blurring_image).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_light):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.blurring[1] = np.array([2.0, 2.0])

            lp = galaxy_light.light_profiles[0]

            lp_blurring_image = lp.intensities_from_grid(imaging_grids.blurring)

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == lp_blurring_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == lp_blurring_image).all()

        def test__same_as_above__use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_blurring_image = lp0.intensities_from_grid(imaging_grids.blurring)
            lp1_blurring_image = lp1.intensities_from_grid(imaging_grids.blurring)

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == lp0_blurring_image + lp1_blurring_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == lp0_blurring_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[1] == lp1_blurring_image).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, imaging_grids, galaxy_light):
            galaxy_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[galaxy_light])

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == galaxy_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_light):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.blurring[1] = np.array([2.0, 2.0])

            galaxy_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[galaxy_light])

            plane = pl.Plane(galaxies=[galaxy_light], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == galaxy_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[g0])
            g1_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[g1])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == g0_image + g1_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == g0_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[1] == g1_image).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_intensities_for_each(self, imaging_grids):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            g0_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[g0])
            g1_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[g1])
            g2_image = pl.intensities_from_grid(imaging_grids.blurring, galaxies=[g2])

            plane = pl.Plane(galaxies=[g0, g1, g2], grids=imaging_grids)

            assert (plane._image_plane_blurring_image == g0_image + g1_image + g2_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[0] == g0_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[1] == g1_image).all()
            assert (plane._image_plane_blurring_images_of_galaxies[2] == g2_image).all()

    class TestSurfaceDensity:

        def test__surface_density_from_plane__same_as_its_mass_profile(self, imaging_grids, galaxy_mass):
            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.surface_density_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_image_pixel_0 = (mp_sub_image[0] + mp_sub_image[1] + mp_sub_image[2] + mp_sub_image[3]) / 4
            mp_image_pixel_1 = (mp_sub_image[4] + mp_sub_image[5] + mp_sub_image[6] + mp_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._surface_density[0] == mp_image_pixel_0).all()
            assert (plane._surface_density[1] == mp_image_pixel_1).all()
            assert (plane._surface_density_of_galaxies[0][0] == mp_image_pixel_0).all()
            assert (plane._surface_density_of_galaxies[0][1] == mp_image_pixel_1).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.surface_density_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_image_pixel_0 = (mp_sub_image[0] + mp_sub_image[1] + mp_sub_image[2] + mp_sub_image[3]) / 4
            mp_image_pixel_1 = (mp_sub_image[4] + mp_sub_image[5] + mp_sub_image[6] + mp_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._surface_density[0] == mp_image_pixel_0).all()
            assert (plane._surface_density[1] == mp_image_pixel_1).all()
            assert (plane._surface_density_of_galaxies[0][0] == mp_image_pixel_0).all()
            assert (plane._surface_density_of_galaxies[0][1] == mp_image_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_image = mp0.surface_density_from_grid(imaging_grids.sub)
            mp1_sub_image = mp1.surface_density_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp0_image_pixel_0 = (mp0_sub_image[0] + mp0_sub_image[1] + mp0_sub_image[2] + mp0_sub_image[3]) / 4
            mp0_image_pixel_1 = (mp0_sub_image[4] + mp0_sub_image[5] + mp0_sub_image[6] + mp0_sub_image[7]) / 4
            mp1_image_pixel_0 = (mp1_sub_image[0] + mp1_sub_image[1] + mp1_sub_image[2] + mp1_sub_image[3]) / 4
            mp1_image_pixel_1 = (mp1_sub_image[4] + mp1_sub_image[5] + mp1_sub_image[6] + mp1_sub_image[7]) / 4

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._surface_density[0] == mp0_image_pixel_0 + mp1_image_pixel_0).all()
            assert (plane._surface_density[1] == mp0_image_pixel_1 + mp1_image_pixel_1).all()
            assert (plane._surface_density_of_galaxies[0][0] == mp0_image_pixel_0).all()
            assert (plane._surface_density_of_galaxies[1][0] == mp1_image_pixel_0).all()
            assert (plane._surface_density_of_galaxies[0][1] == mp0_image_pixel_1).all()
            assert (plane._surface_density_of_galaxies[1][1] == mp1_image_pixel_1).all()

        def test__surface_density__same_as_its_galaxy(self, imaging_grids, galaxy_mass):
            galaxy_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._surface_density == galaxy_surface_density).all()
            assert (plane._surface_density_of_galaxies[0] == galaxy_surface_density).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            galaxy_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._surface_density == galaxy_surface_density).all()
            assert (plane._surface_density_of_galaxies[0] == galaxy_surface_density).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[g1])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._surface_density == g0_surface_density + g1_surface_density).all()
            assert (plane._surface_density_of_galaxies[0] == g0_surface_density).all()
            assert (plane._surface_density_of_galaxies[1] == g1_surface_density).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_surface_density_for_each(self, imaging_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[g1])
            g2_surface_density = pl.surface_density_from_grid(imaging_grids.sub, galaxies=[g2])

            plane = pl.Plane(galaxies=[g0, g1, g2], grids=imaging_grids)

            assert (plane._surface_density == g0_surface_density + g1_surface_density + g2_surface_density).all()
            assert (plane._surface_density_of_galaxies[0] == g0_surface_density).all()
            assert (plane._surface_density_of_galaxies[1] == g1_surface_density).all()
            assert (plane._surface_density_of_galaxies[2] == g2_surface_density).all()

    class TestPotential:

        def test__potential_from_plane__same_as_its_mass_profile(self, imaging_grids, galaxy_mass):
            mp = galaxy_mass.mass_profiles[0]

            mp_sub_potential = mp.potential_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_potential_pixel_0 = (mp_sub_potential[0] + mp_sub_potential[1] + mp_sub_potential[2] + mp_sub_potential
                [3]) / 4
            mp_potential_pixel_1 = (mp_sub_potential[4] + mp_sub_potential[5] + mp_sub_potential[6] + mp_sub_potential
                [7]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._potential[0] == mp_potential_pixel_0).all()
            assert (plane._potential[1] == mp_potential_pixel_1).all()
            assert (plane._potential_of_galaxies[0][0] == mp_potential_pixel_0).all()
            assert (plane._potential_of_galaxies[0][1] == mp_potential_pixel_1).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_potential = mp.potential_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_potential_pixel_0 = (mp_sub_potential[0] + mp_sub_potential[1] + mp_sub_potential[2] + mp_sub_potential
                [3]) / 4
            mp_potential_pixel_1 = (mp_sub_potential[4] + mp_sub_potential[5] + mp_sub_potential[6] + mp_sub_potential
                [7]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._potential[0] == mp_potential_pixel_0).all()
            assert (plane._potential[1] == mp_potential_pixel_1).all()
            assert (plane._potential_of_galaxies[0][0] == mp_potential_pixel_0).all()
            assert (plane._potential_of_galaxies[0][1] == mp_potential_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_potential = mp0.potential_from_grid(imaging_grids.sub)
            mp1_sub_potential = mp1.potential_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp0_potential_pixel_0 = (mp0_sub_potential[0] + mp0_sub_potential[1] + mp0_sub_potential[2] + mp0_sub_potential[3]) / 4
            mp0_potential_pixel_1 = (mp0_sub_potential[4] + mp0_sub_potential[5] + mp0_sub_potential[6] + mp0_sub_potential[7]) / 4
            mp1_potential_pixel_0 = (mp1_sub_potential[0] + mp1_sub_potential[1] + mp1_sub_potential[2] + mp1_sub_potential[3]) / 4
            mp1_potential_pixel_1 = (mp1_sub_potential[4] + mp1_sub_potential[5] + mp1_sub_potential[6] + mp1_sub_potential[7]) / 4

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._potential[0] == mp0_potential_pixel_0 + mp1_potential_pixel_0).all()
            assert (plane._potential[1] == mp0_potential_pixel_1 + mp1_potential_pixel_1).all()
            assert (plane._potential_of_galaxies[0][0] == mp0_potential_pixel_0).all()
            assert (plane._potential_of_galaxies[1][0] == mp1_potential_pixel_0).all()
            assert (plane._potential_of_galaxies[0][1] == mp0_potential_pixel_1).all()
            assert (plane._potential_of_galaxies[1][1] == mp1_potential_pixel_1).all()

        def test__potential__same_as_its_galaxy(self, imaging_grids, galaxy_mass):
            galaxy_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._potential == galaxy_potential).all()
            assert (plane._potential_of_galaxies[0] == galaxy_potential).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            galaxy_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._potential == galaxy_potential).all()
            assert (plane._potential_of_galaxies[0] == galaxy_potential).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[g1])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._potential == g0_potential + g1_potential).all()
            assert (plane._potential_of_galaxies[0] == g0_potential).all()
            assert (plane._potential_of_galaxies[1] == g1_potential).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_potential_for_each(self, imaging_grids):
            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[g1])
            g2_potential = pl.potential_from_grid(imaging_grids.sub, galaxies=[g2])

            plane = pl.Plane(galaxies=[g0, g1, g2], grids=imaging_grids)

            assert (plane._potential == g0_potential + g1_potential + g2_potential).all()
            assert (plane._potential_of_galaxies[0] == g0_potential).all()
            assert (plane._potential_of_galaxies[1] == g1_potential).all()
            assert (plane._potential_of_galaxies[2] == g2_potential).all()

    class TestDeflections:

        def test__deflections_from_plane__same_as_its_mass_profile(self, imaging_grids, galaxy_mass):

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.deflections_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_image_pixel_0x = (mp_sub_image[0 ,0] + mp_sub_image[1 ,0] + mp_sub_image[2 ,0] + mp_sub_image[3 ,0]) / 4
            mp_image_pixel_1x = (mp_sub_image[4 ,0] + mp_sub_image[5 ,0] + mp_sub_image[6 ,0] + mp_sub_image[7 ,0]) / 4
            mp_image_pixel_0y = (mp_sub_image[0 ,1] + mp_sub_image[1 ,1] + mp_sub_image[2 ,1] + mp_sub_image[3 ,1]) / 4
            mp_image_pixel_1y = (mp_sub_image[4 ,1] + mp_sub_image[5 ,1] + mp_sub_image[6 ,1] + mp_sub_image[7 ,1]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._deflections[0 ,0] == mp_image_pixel_0x).all()
            assert (plane._deflections[0 ,1] == mp_image_pixel_0y).all()
            assert (plane._deflections[1 ,0] == mp_image_pixel_1x).all()
            assert (plane._deflections[1 ,1] == mp_image_pixel_1y).all()
            assert (plane._deflections_of_galaxies[0][0 ,0] == mp_image_pixel_0x).all()
            assert (plane._deflections_of_galaxies[0][0 ,1] == mp_image_pixel_0y).all()
            assert (plane._deflections_of_galaxies[0][1 ,0] == mp_image_pixel_1x).all()
            assert (plane._deflections_of_galaxies[0][1 ,1] == mp_image_pixel_1y).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.deflections_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp_image_pixel_0x = (mp_sub_image[0 ,0] + mp_sub_image[1 ,0] + mp_sub_image[2 ,0] + mp_sub_image[3 ,0]) / 4
            mp_image_pixel_1x = (mp_sub_image[4 ,0] + mp_sub_image[5 ,0] + mp_sub_image[6 ,0] + mp_sub_image[7 ,0]) / 4
            mp_image_pixel_0y = (mp_sub_image[0 ,1] + mp_sub_image[1 ,1] + mp_sub_image[2 ,1] + mp_sub_image[3 ,1]) / 4
            mp_image_pixel_1y = (mp_sub_image[4 ,1] + mp_sub_image[5 ,1] + mp_sub_image[6 ,1] + mp_sub_image[7 ,1]) / 4

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._deflections[0 ,0] == mp_image_pixel_0x).all()
            assert (plane._deflections[0 ,1] == mp_image_pixel_0y).all()
            assert (plane._deflections[1 ,0] == mp_image_pixel_1x).all()
            assert (plane._deflections[1 ,1] == mp_image_pixel_1y).all()
            assert (plane._deflections_of_galaxies[0][0 ,0] == mp_image_pixel_0x).all()
            assert (plane._deflections_of_galaxies[0][0 ,1] == mp_image_pixel_0y).all()
            assert (plane._deflections_of_galaxies[0][1 ,0] == mp_image_pixel_1x).all()
            assert (plane._deflections_of_galaxies[0][1 ,1] == mp_image_pixel_1y).all()

        def test__same_as_above__use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_image = mp0.deflections_from_grid(imaging_grids.sub)
            mp1_sub_image = mp1.deflections_from_grid(imaging_grids.sub)

            # Perform sub gridding average manually
            mp0_image_pixel_0x = (mp0_sub_image[0 ,0] + mp0_sub_image[1 ,0] + mp0_sub_image[2 ,0] + mp0_sub_image
                [3 ,0]) / 4
            mp0_image_pixel_1x = (mp0_sub_image[4 ,0] + mp0_sub_image[5 ,0] + mp0_sub_image[6 ,0] + mp0_sub_image
                [7 ,0]) / 4
            mp0_image_pixel_0y = (mp0_sub_image[0 ,1] + mp0_sub_image[1 ,1] + mp0_sub_image[2 ,1] + mp0_sub_image
                [3 ,1]) / 4
            mp0_image_pixel_1y = (mp0_sub_image[4 ,1] + mp0_sub_image[5 ,1] + mp0_sub_image[6 ,1] + mp0_sub_image
                [7 ,1]) / 4

            mp1_image_pixel_0x = (mp1_sub_image[0 ,0] + mp1_sub_image[1 ,0] + mp1_sub_image[2 ,0] + mp1_sub_image
                [3 ,0]) / 4
            mp1_image_pixel_1x = (mp1_sub_image[4 ,0] + mp1_sub_image[5 ,0] + mp1_sub_image[6 ,0] + mp1_sub_image
                [7 ,0]) / 4
            mp1_image_pixel_0y = (mp1_sub_image[0 ,1] + mp1_sub_image[1 ,1] + mp1_sub_image[2 ,1] + mp1_sub_image
                [3 ,1]) / 4
            mp1_image_pixel_1y = (mp1_sub_image[4 ,1] + mp1_sub_image[5 ,1] + mp1_sub_image[6 ,1] + mp1_sub_image
                [7 ,1]) / 4

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._deflections[0 ,0] == mp0_image_pixel_0x + mp1_image_pixel_0x).all()
            assert (plane._deflections[1 ,0] == mp0_image_pixel_1x + mp1_image_pixel_1x).all()
            assert (plane._deflections[0 ,1] == mp0_image_pixel_0y + mp1_image_pixel_0y).all()
            assert (plane._deflections[1 ,1] == mp0_image_pixel_1y + mp1_image_pixel_1y).all()
            assert (plane._deflections_of_galaxies[0][0 ,0] == mp0_image_pixel_0x).all()
            assert (plane._deflections_of_galaxies[0][0 ,1] == mp0_image_pixel_0y).all()
            assert (plane._deflections_of_galaxies[0][1 ,0] == mp0_image_pixel_1x).all()
            assert (plane._deflections_of_galaxies[0][1 ,1] == mp0_image_pixel_1y).all()
            assert (plane._deflections_of_galaxies[1][0 ,0] == mp1_image_pixel_0x).all()
            assert (plane._deflections_of_galaxies[1][0 ,1] == mp1_image_pixel_0y).all()
            assert (plane._deflections_of_galaxies[1][1 ,0] == mp1_image_pixel_1x).all()
            assert (plane._deflections_of_galaxies[1][1 ,1] == mp1_image_pixel_1y).all()

        def test__deflections__same_as_its_galaxy(self, imaging_grids, galaxy_mass):

            galaxy_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._deflections == galaxy_deflections).all()
            assert (plane._deflections_of_galaxies[0] == galaxy_deflections).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, imaging_grids, galaxy_mass):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            galaxy_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[galaxy_mass])

            plane = pl.Plane(galaxies=[galaxy_mass], grids=imaging_grids)

            assert (plane._deflections == galaxy_deflections).all()
            assert (plane._deflections_of_galaxies[0] == galaxy_deflections).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, imaging_grids):
            # Overwrite one value so intensity in each pixel is different
            imaging_grids.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[g1])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            assert (plane._deflections == g0_deflections + g1_deflections).all()
            assert (plane._deflections_of_galaxies[0] == g0_deflections).all()
            assert (plane._deflections_of_galaxies[1] == g1_deflections).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_deflections_for_each(self, imaging_grids):
            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[g0])
            g1_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[g1])
            g2_deflections = pl.deflections_from_grid(imaging_grids.sub, galaxies=[g2])

            plane = pl.Plane(galaxies=[g0, g1, g2], grids=imaging_grids)

            assert (plane._deflections == g0_deflections + g1_deflections + g2_deflections).all()
            assert (plane._deflections_of_galaxies[0] == g0_deflections).all()
            assert (plane._deflections_of_galaxies[1] == g1_deflections).all()
            assert (plane._deflections_of_galaxies[2] == g2_deflections).all()

    class TestPlaneImage:

        def test__shape_3x3__image_of_plane__same_as_light_profile_on_identical_uniform_grid(self, imaging_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            g0_image = g0.intensities_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                               [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))
            g0_image = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(g0_image, shape=(3 ,3))

            imaging_grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)

            plane_image = plane.plane_image(shape=(3, 3))
            assert plane_image == pytest.approx(g0_image, 1e-4)
            assert (plane_image.grid == imaging_grids.image).all()

        def test__different_shape_and_multiple_galaxies(self, imaging_grids):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))

            g0_image = g0.intensities_from_grid(grid=np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                               [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))
            g0_image = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(g0_image, shape=(2 ,3))

            g1_image = g1.intensities_from_grid(grid=np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                               [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))
            g1_image = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(g1_image, shape=(2 ,3))

            imaging_grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            plane = pl.Plane(galaxies=[g0, g1], grids=imaging_grids)

            plane_image = plane.plane_image(shape=(2, 3))
            assert plane_image == pytest.approx(g0_image + g1_image, 1e-4)
            assert (plane_image.grid == imaging_grids.image).all()

        def test__ensure_index_of_plane_image_has_negative_arcseconds_at_start(self, imaging_grids):
            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The centre -1.6, -1.6 of the galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane _image.

            imaging_grids.image = mask.ImageGrid(np.array([[-2.0, -2.0], [2.0, 2.0]]), shape_2d=(5, 5),
                                                 grid_to_pixel=None)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, -1.6), intensity=1.0))
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            plane_image = plane.plane_image(shape=(5 ,5))

            assert plane_image.shape == (5, 5)
            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (0, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0))
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            plane_image = plane.plane_image(shape=(5 ,5))
            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (0, 4)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0))
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            plane_image = plane.plane_image(shape=(5 ,5))
            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (4, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0))
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            plane_image = plane.plane_image(shape=(5 ,5))
            assert np.unravel_index(plane_image.argmax(), plane_image.shape) == (4, 4)

    class TestXYTicksOfPlane:

        def test__compute_xticks_from_image_grid_correctly__are_rounded_to_2dp(self, imaging_grids):
            g0 = g.Galaxy()

            imaging_grids.image = mask.ImageGrid(np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.xticks == pytest.approx(np.array([-0.3, -0.1, 0.1, 0.3]), 1e-3)

            imaging_grids.image = mask.ImageGrid(np.array([[-6.0, -10.5], [6.0, 0.5], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.xticks == pytest.approx(np.array([-6.0, -2.0, 2.0, 6.0]), 1e-3)

            imaging_grids.image = mask.ImageGrid(np.array([[-1.0, -0.5], [1.0, 0.5], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.xticks == pytest.approx(np.array([-1.0, -0.33, 0.33, 1.0]), 1e-3)

        def test__compute_yticks_from_image_grid_correctly__are_rounded_to_2dp(self, imaging_grids):
            g0 = g.Galaxy()

            imaging_grids.image = mask.ImageGrid(np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.yticks == pytest.approx(np.array([-0.3, -0.1, 0.1, 0.3]), 1e-3)

            imaging_grids.image = mask.ImageGrid(np.array([[-10.5, -6.0], [0.5, 6.0], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.yticks == pytest.approx(np.array([-6.0, -2.0, 2.0, 6.0]), 1e-3)

            imaging_grids.image = mask.ImageGrid(np.array([[-0.5, -1.0], [0.5, 1.0], [0.3, 0.3], [-0.3, -0.3]]),
                                                 shape_2d=(3, 3), grid_to_pixel=None)
            plane = pl.Plane(galaxies=[g0], grids=imaging_grids)
            assert plane.yticks == pytest.approx(np.array([-1.0, -0.33, 0.33, 1.0]), 1e-3)

    class TestPixeizationMapper:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, imaging_grids):
            galaxy_no_pix = g.Galaxy()

            plane = pl.Plane(galaxies=[galaxy_no_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.mapper is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, imaging_grids):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane = pl.Plane(galaxies=[galaxy_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.mapper == 1

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, imaging_grids):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane = pl.Plane(galaxies=[galaxy_no_pix, galaxy_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.mapper == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, imaging_grids):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane = pl.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1], grids=imaging_grids, borders=MockBorders())

            with pytest.raises(exc.PixelizationException):
                plane.mapper

    class TestRegularization:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, imaging_grids):
            galaxy_no_pix = g.Galaxy()

            plane = pl.Plane(galaxies=[galaxy_no_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.regularization is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, imaging_grids):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane = pl.Plane(galaxies=[galaxy_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.regularization.value == 0

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, imaging_grids):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane = pl.Plane(galaxies=[galaxy_no_pix, galaxy_pix], grids=imaging_grids, borders=MockBorders())

            assert plane.regularization.value == 0

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, imaging_grids):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane = pl.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1], grids=imaging_grids, borders=MockBorders())

            with pytest.raises(exc.PixelizationException):
                plane.regularization