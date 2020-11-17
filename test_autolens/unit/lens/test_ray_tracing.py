import autolens as al
import numpy as np
import pytest
import os
from os import path
import shutil
from astropy import cosmology as cosmo
from skimage import measure
from autoarray.mock import mock as mock_inv


test_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "tracer"
)


def critical_curve_via_magnification_from_tracer_and_grid(tracer, grid):
    magnification = tracer.magnification_from_grid(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.in_2d, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = (
            grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
                grid_pixels_1d=pixel_coord, shape_2d=magnification.sub_shape_2d
            )
        )

        critical_curve = np.array(grid=critical_curve)

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from_tracer_and_grid(tracer, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from_tracer_and_grid(
        tracer=tracer, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = tracer.deflections_from_grid(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


class TestAbstractTracer:
    class TestProperties:
        def test__total_planes(self):

            tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5)])

            assert tracer.total_planes == 1

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
            )

            assert tracer.total_planes == 2

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    al.Galaxy(redshift=1.0),
                    al.Galaxy(redshift=2.0),
                    al.Galaxy(redshift=3.0),
                ]
            )

            assert tracer.total_planes == 3

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    al.Galaxy(redshift=1.0),
                    al.Galaxy(redshift=2.0),
                    al.Galaxy(redshift=1.0),
                ]
            )

            assert tracer.total_planes == 2

        def test__has_galaxy_with_light_profile(self):

            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_mp = al.Galaxy(redshift=0.5, mass_profile=al.mp.SphericalIsothermal())

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.has_light_profile is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_mp, gal_mp])

            assert tracer.has_light_profile is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_lp])

            assert tracer.has_light_profile is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal])

            assert tracer.has_light_profile is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_mp])

            assert tracer.has_light_profile is True

        def test_plane_with_galaxy(self, sub_grid_7x7):

            g1 = al.Galaxy(redshift=1)
            g2 = al.Galaxy(redshift=2)

            tracer = al.Tracer.from_galaxies(galaxies=[g1, g2])

            assert tracer.plane_with_galaxy(g1).galaxies == [g1]
            assert tracer.plane_with_galaxy(g2).galaxies == [g2]

        def test__has_galaxy_with_mass_profile(self, sub_grid_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_mp = al.Galaxy(redshift=0.5, mass_profile=al.mp.SphericalIsothermal())

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.has_mass_profile is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_mp, gal_mp])

            assert tracer.has_mass_profile is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_lp])

            assert tracer.has_mass_profile is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal])

            assert tracer.has_mass_profile is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_mp])

            assert tracer.has_mass_profile is True

        def test__planes_indexes_with_inversion(self):

            gal = al.Galaxy(redshift=0.5)
            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.plane_indexes_with_pixelizations == []

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.plane_indexes_with_pixelizations == [0]

            gal_pix = al.Galaxy(
                redshift=1.0,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.plane_indexes_with_pixelizations == [1]

            gal_pix_0 = al.Galaxy(
                redshift=0.6,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            gal_pix_1 = al.Galaxy(
                redshift=2.0,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            gal0 = al.Galaxy(redshift=0.25)
            gal1 = al.Galaxy(redshift=0.5)
            gal2 = al.Galaxy(redshift=0.75)

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal_pix_0, gal_pix_1, gal0, gal1, gal2]
            )

            assert tracer.plane_indexes_with_pixelizations == [2, 4]

        def test__has_galaxy_with_pixelization(self, sub_grid_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.has_pixelization is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_lp])

            assert tracer.has_pixelization is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal_pix])

            assert tracer.has_pixelization is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.has_pixelization is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal_lp])

            assert tracer.has_pixelization is True

        def test__has_galaxy_with_regularization(self, sub_grid_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_reg = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.has_regularization is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_lp])

            assert tracer.has_regularization is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_reg, gal_reg])

            assert tracer.has_regularization is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_reg, gal])

            assert tracer.has_regularization is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_reg, gal_lp])

            assert tracer.has_regularization is True

        def test__has_galaxy_with_hyper_galaxy(self, sub_grid_7x7):

            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_hyper = al.Galaxy(redshift=0.5, hyper_galaxy=al.HyperGalaxy())

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.has_hyper_galaxy is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_lp, gal_lp])

            assert tracer.has_hyper_galaxy is False

            tracer = al.Tracer.from_galaxies(galaxies=[gal_hyper, gal_hyper])

            assert tracer.has_hyper_galaxy is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_hyper, gal])

            assert tracer.has_hyper_galaxy is True

            tracer = al.Tracer.from_galaxies(galaxies=[gal_hyper, gal_lp])

            assert tracer.has_hyper_galaxy is True

        def test__upper_plane_index_with_light_profile(self):

            g0 = al.Galaxy(redshift=0.5)
            g1 = al.Galaxy(redshift=1.0)
            g2 = al.Galaxy(redshift=2.0)
            g3 = al.Galaxy(redshift=3.0)

            g0_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            g1_lp = al.Galaxy(redshift=1.0, light_profile=al.lp.LightProfile())
            g2_lp = al.Galaxy(redshift=2.0, light_profile=al.lp.LightProfile())
            g3_lp = al.Galaxy(redshift=3.0, light_profile=al.lp.LightProfile())

            tracer = al.Tracer.from_galaxies(galaxies=[g0_lp])

            assert tracer.upper_plane_index_with_light_profile == 0

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g0_lp])

            assert tracer.upper_plane_index_with_light_profile == 0

            tracer = al.Tracer.from_galaxies(galaxies=[g1_lp])

            assert tracer.upper_plane_index_with_light_profile == 0

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1_lp])

            assert tracer.upper_plane_index_with_light_profile == 1

            tracer = al.Tracer.from_galaxies(galaxies=[g0_lp, g1_lp, g2_lp])

            assert tracer.upper_plane_index_with_light_profile == 2

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2_lp])

            assert tracer.upper_plane_index_with_light_profile == 2

            tracer = al.Tracer.from_galaxies(galaxies=[g0_lp, g1, g2, g3_lp])

            assert tracer.upper_plane_index_with_light_profile == 3

            tracer = al.Tracer.from_galaxies(galaxies=[g0_lp, g1, g2_lp, g3])

            assert tracer.upper_plane_index_with_light_profile == 2

        def test__hyper_model_image_of_plane_with_pixelization(self, sub_grid_7x7):

            gal = al.Galaxy(redshift=0.5)
            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.hyper_galaxy_image_of_planes_with_pixelizations == [None]

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal_pix])

            assert tracer.hyper_galaxy_image_of_planes_with_pixelizations == [None]

            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
                hyper_galaxy_image=1,
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.hyper_galaxy_image_of_planes_with_pixelizations == [1]

            gal0 = al.Galaxy(redshift=0.25)
            gal1 = al.Galaxy(redshift=0.75)
            gal2 = al.Galaxy(redshift=1.5)

            gal_pix0 = al.Galaxy(
                redshift=0.5,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
                hyper_galaxy_image=1,
            )

            gal_pix1 = al.Galaxy(
                redshift=2.0,
                pixelization=al.pix.Pixelization(),
                regularization=al.reg.Constant(),
                hyper_galaxy_image=2,
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal0, gal1, gal2, gal_pix0, gal_pix1]
            )

            assert tracer.hyper_galaxy_image_of_planes_with_pixelizations == [
                None,
                1,
                None,
                None,
                2,
            ]

    class TestPixelizations:
        def test__no_galaxy_has_regularization__returns_list_of_ones(
            self, sub_grid_7x7
        ):
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

            assert tracer.pixelizations_of_planes == [None]

        def test__source_galaxy_has_regularization__returns_list_with_none_and_regularization(
            self, sub_grid_7x7
        ):
            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            assert tracer.pixelizations_of_planes[0] is None
            assert tracer.pixelizations_of_planes[1].value == 1

        def test__both_galaxies_have_pixelization__returns_both_pixelizations(
            self, sub_grid_7x7
        ):
            galaxy_pix_0 = al.Galaxy(
                redshift=0.5,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(3, 3)),
            )

            galaxy_pix_1 = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(value=2),
                regularization=mock_inv.MockRegularization(matrix_shape=(4, 4)),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_pix_0, galaxy_pix_1])

            assert tracer.pixelizations_of_planes[0].value == 1
            assert tracer.pixelizations_of_planes[1].value == 2

    class TestRegularizations:
        def test__no_galaxy_has_regularization__returns_empty_list(self, sub_grid_7x7):
            galaxy_no_reg = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_no_reg])

            assert tracer.regularizations_of_planes == [None]

        def test__source_galaxy_has_regularization__returns_regularizations(
            self, sub_grid_7x7
        ):
            galaxy_reg = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_reg = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_reg])

            assert tracer.regularizations_of_planes[0] is None
            assert tracer.regularizations_of_planes[1].shape == (1, 1)

        def test__both_galaxies_have_regularization__returns_both_regularizations(
            self, sub_grid_7x7
        ):
            galaxy_reg_0 = al.Galaxy(
                redshift=0.5,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(3, 3)),
            )

            galaxy_reg_1 = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(value=2),
                regularization=mock_inv.MockRegularization(matrix_shape=(4, 4)),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_reg_0, galaxy_reg_1])

            assert tracer.regularizations_of_planes[0].shape == (3, 3)
            assert tracer.regularizations_of_planes[1].shape == (4, 4)

    class TestGalaxyLists:
        def test__galaxy_list__comes_in_plane_redshift_order(self, sub_grid_7x7):
            g0 = al.Galaxy(redshift=0.5)
            g1 = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            assert tracer.galaxies == [g0, g1]

            g2 = al.Galaxy(redshift=1.0)
            g3 = al.Galaxy(redshift=1.0)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, g3])

            assert tracer.galaxies == [g0, g1, g2, g3]

            g4 = al.Galaxy(redshift=0.75)
            g5 = al.Galaxy(redshift=1.5)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, g3, g4, g5])

            assert tracer.galaxies == [g0, g1, g4, g2, g3, g5]

        # def test__galaxy_in_planes_lists__comes_in_lists_of_planes_in_redshift_order(self, sub_grid_7x7):
        #     g0 = al.Galaxy(redshift=0.5)
        #     g1 = al.Galaxy(redshift=0.5)
        #
        #     tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])
        #
        #     assert tracer.galaxies_in_planes == [[g0, g1]]
        #
        #     g2 = al.Galaxy(redshift=1.0)
        #     g3 = al.Galaxy(redshift=1.0)
        #
        #     tracer = al.Tracer.from_galaxies(galaxies=[g0, g1], galaxies=[g2, g3],
        #                                                  )
        #
        #     assert tracer.galaxies_in_planes == [[g0, g1], [g2, g3]]
        #
        #     g4 = al.Galaxy(redshift=0.75)
        #     g5 = al.Galaxy(redshift=1.5)
        #
        #     tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, g3, g4, g5],
        #                                            )
        #
        #     assert tracer.galaxies_in_planes == [[g0, g1], [g4], [g2, g3], [g5]]

    class TestLightProfileQuantities:
        def test__extract_centres_of_all_light_profiles_of_all_planes_and_galaxies(
            self,
        ):
            g0 = al.Galaxy(
                redshift=0.5, light=al.lp.SphericalGaussian(centre=(1.0, 1.0))
            )
            g1 = al.Galaxy(
                redshift=0.5, light=al.lp.SphericalGaussian(centre=(2.0, 2.0))
            )
            g2 = al.Galaxy(
                redshift=1.0,
                light0=al.lp.SphericalGaussian(centre=(3.0, 3.0)),
                light1=al.lp.SphericalGaussian(centre=(4.0, 4.0)),
            )

            plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
            plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.light_profile_centres == []
            assert tracer.light_profile_centres == []

            plane_0 = al.Plane(galaxies=[g0], redshift=None)
            plane_1 = al.Plane(galaxies=[g1], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.light_profile_centres.in_list == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
            plane_1 = al.Plane(galaxies=[g2], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.light_profile_centres.in_list == [
                [(1.0, 1.0), (2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

    class TestMassProfileQuantities:
        def test__extract_mass_profiles_of_all_planes_and_galaxies(self):
            g0 = al.Galaxy(
                redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(1.0, 1.0))
            )
            g1 = al.Galaxy(
                redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0))
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass0=al.mp.SphericalIsothermal(centre=(3.0, 3.0)),
                mass1=al.mp.SphericalIsothermal(centre=(4.0, 4.0)),
            )

            plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
            plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profiles == []
            assert tracer.mass_profiles == []

            plane_0 = al.Plane(galaxies=[g0], redshift=None)
            plane_1 = al.Plane(galaxies=[g1], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profiles_of_planes == [[g0.mass], [g1.mass]]
            assert tracer.mass_profiles == [g0.mass, g1.mass]

            plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
            plane_1 = al.Plane(galaxies=[g2], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profiles_of_planes == [
                [g0.mass, g1.mass],
                [g2.mass0, g2.mass1],
            ]
            assert tracer.mass_profiles == [g0.mass, g1.mass, g2.mass0, g2.mass1]

        def test__extract_centres_of_all_mass_profiles_of_all_planes_and_galaxies__ignores_mass_sheets(
            self,
        ):
            g0 = al.Galaxy(
                redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(1.0, 1.0))
            )
            g1 = al.Galaxy(
                redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0))
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass0=al.mp.SphericalIsothermal(centre=(3.0, 3.0)),
                mass1=al.mp.SphericalIsothermal(centre=(4.0, 4.0)),
            )

            plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
            plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profile_centres == []

            plane_0 = al.Plane(galaxies=[g0], redshift=None)
            plane_1 = al.Plane(galaxies=[g1], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profile_centres.in_list == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
            plane_1 = al.Plane(galaxies=[g2], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profile_centres.in_list == [
                [(1.0, 1.0), (2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

            g1 = al.Galaxy(
                redshift=0.5,
                mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0)),
                sheet=al.mp.MassSheet(centre=(10.0, 10.0)),
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass0=al.mp.SphericalIsothermal(centre=(3.0, 3.0)),
                mass1=al.mp.SphericalIsothermal(centre=(4.0, 4.0)),
                sheet=al.mp.MassSheet(centre=(10.0, 10.0)),
            )

            plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
            plane_1 = al.Plane(galaxies=[g2], redshift=None)

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

            assert tracer.mass_profile_centres.in_list == [
                [(1.0, 1.0), (2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

    class TestPickle:
        def test__tracer_can_be_pickled_and_loaded(self):

            if path.exists(test_path):
                shutil.rmtree(test_path)

            if not path.exists(test_path):
                os.mkdir(test_path)

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.1))
                ]
            )

            tracer.save(file_path=test_path, filename="test_tracer")

            tracer = al.Tracer.load(file_path=test_path, filename="test_tracer")

            assert tracer.galaxies[0].light.intensity == 1.1


class TestAbstractTracerLensing:
    class TestTracedGridsFromGrid:
        def test__x2_planes__no_galaxy__image_and_source_planes_setup__same_coordinates(
            self, sub_grid_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.25, -1.25]), 1e-3
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.25, -0.75]), 1e-3
            )
            assert traced_grids_of_planes[0][2] == pytest.approx(
                np.array([0.75, -1.25]), 1e-3
            )
            assert traced_grids_of_planes[0][3] == pytest.approx(
                np.array([0.75, -0.75]), 1e-3
            )

            assert traced_grids_of_planes[1][0] == pytest.approx(
                np.array([1.25, -1.25]), 1e-3
            )
            assert traced_grids_of_planes[1][1] == pytest.approx(
                np.array([1.25, -0.75]), 1e-3
            )
            assert traced_grids_of_planes[1][2] == pytest.approx(
                np.array([0.75, -1.25]), 1e-3
            )
            assert traced_grids_of_planes[1][3] == pytest.approx(
                np.array([0.75, -0.75]), 1e-3
            )

        def test__x2_planes__sis_lens__traced_grid_includes_deflections__on_planes_setup(
            self, sub_grid_7x7_simple, gal_x1_mp
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal_x1_mp, al.Galaxy(redshift=1.0)]
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7_simple
            )

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )
            assert traced_grids_of_planes[0][2] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][3] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )

            assert traced_grids_of_planes[1][0] == pytest.approx(
                np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3
            )
            assert traced_grids_of_planes[1][1] == pytest.approx(
                np.array([0.0, 0.0]), 1e-3
            )
            assert traced_grids_of_planes[1][2] == pytest.approx(
                np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3
            )
            assert traced_grids_of_planes[1][3] == pytest.approx(
                np.array([0.0, 0.0]), 1e-3
            )

        def test__same_as_above_but_2_sis_lenses__deflections_double(
            self, sub_grid_7x7_simple, gal_x1_mp
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal_x1_mp, gal_x1_mp, al.Galaxy(redshift=1.0)]
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7_simple
            )

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )
            assert traced_grids_of_planes[0][2] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][3] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )

            assert traced_grids_of_planes[1][0] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
            )
            assert traced_grids_of_planes[1][1] == pytest.approx(
                np.array([-1.0, 0.0]), 1e-3
            )
            assert traced_grids_of_planes[1][2] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
            )

            assert traced_grids_of_planes[1][3] == pytest.approx(
                np.array([-1.0, 0.0]), 1e-3
            )

        def test__4_planes__grids_are_correct__sis_mass_profile(
            self, sub_grid_7x7_simple
        ):

            g0 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g2 = al.Galaxy(
                redshift=0.1,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g3 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g4 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g5 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4, g5], cosmology=cosmo.Planck15
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7_simple
            )

            # The scaling factors are as follows and were computed independently from the test_autoarray.
            beta_01 = 0.9348
            beta_02 = 0.9839601
            # Beta_03 = 1.0
            beta_12 = 0.7539734
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = np.sqrt(2) / 2.0

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.0, 1.0]), 1e-4
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-4
            )

            assert traced_grids_of_planes[1][0] == pytest.approx(
                np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
            )
            assert traced_grids_of_planes[1][1] == pytest.approx(
                np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
            )

            defl11 = g0.deflections_from_grid(
                grid=np.array([[(1.0 - beta_01 * val), (1.0 - beta_01 * val)]])
            )
            defl12 = g0.deflections_from_grid(
                grid=np.array([[(1.0 - beta_01 * 1.0), 0.0]])
            )

            assert traced_grids_of_planes[2][0] == pytest.approx(
                np.array(
                    [
                        (1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                        (1.0 - beta_02 * val - beta_12 * defl11[0, 1]),
                    ]
                ),
                1e-4,
            )
            assert traced_grids_of_planes[2][1] == pytest.approx(
                np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12[0, 0]), 0.0]), 1e-4
            )

            assert traced_grids_of_planes[3][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-4
            )

        def test__same_as_above_but_multiple_sets_of_positions(self):
            import math

            g0 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g2 = al.Galaxy(
                redshift=0.1,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g3 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g4 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g5 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4, g5], cosmology=cosmo.Planck15
            )

            traced_positions_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=al.GridCoordinates([[(1.0, 1.0), (1.0, 1.0)], [(1.0, 1.0)]])
            )

            # From unit test_autoarray below:
            # Beta_01 = 0.9348
            beta_02 = 0.9839601
            # Beta_03 = 1.0
            beta_12 = 0.7539734
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert traced_positions_of_planes[0].in_list[0][0] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

            assert traced_positions_of_planes[1].in_list[0][0] == pytest.approx(
                ((1.0 - 0.9348 * val), (1.0 - 0.9348 * val)), 1e-4
            )

            defl11 = g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]])
            )

            assert traced_positions_of_planes[2].in_list[0][0] == pytest.approx(
                (
                    (
                        1.0 - beta_02 * val - beta_12 * defl11[0, 0],
                        1.0 - beta_02 * val - beta_12 * defl11[0, 1],
                    )
                ),
                1e-4,
            )

            assert traced_positions_of_planes[3].in_list[0][0] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

            assert traced_positions_of_planes[0].in_list[0][1] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

            assert traced_positions_of_planes[1].in_list[0][1] == pytest.approx(
                ((1.0 - 0.9348 * val), (1.0 - 0.9348 * val)), 1e-4
            )

            defl11 = g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]])
            )

            assert traced_positions_of_planes[2].in_list[0][1] == pytest.approx(
                (
                    (
                        1.0 - beta_02 * val - beta_12 * defl11[0, 0],
                        1.0 - beta_02 * val - beta_12 * defl11[0, 1],
                    )
                ),
                1e-4,
            )

            assert traced_positions_of_planes[3].in_list[0][1] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

            assert traced_positions_of_planes[0].in_list[1][0] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

            assert traced_positions_of_planes[1].in_list[1][0] == pytest.approx(
                ((1.0 - 0.9348 * val), (1.0 - 0.9348 * val)), 1e-4
            )

            defl11 = g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]])
            )

            assert traced_positions_of_planes[2].in_list[1][0] == pytest.approx(
                (
                    (
                        1.0 - beta_02 * val - beta_12 * defl11[0, 0],
                        1.0 - beta_02 * val - beta_12 * defl11[0, 1],
                    )
                ),
                1e-4,
            )

            assert traced_positions_of_planes[3].in_list[1][0] == pytest.approx(
                (1.0, 1.0), 1e-4
            )

        def test__positions_are_same_as_grid(self):

            g0 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g2 = al.Galaxy(
                redshift=0.1,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g3 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g4 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g5 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4, g5], cosmology=cosmo.Planck15
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )

            traced_positions_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=al.GridCoordinates([[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]])
            )

            assert traced_positions_of_planes[0].in_list[0][0] == tuple(
                traced_grids_of_planes[0][0]
            )

            assert traced_positions_of_planes[1].in_list[0][0] == tuple(
                traced_grids_of_planes[1][0]
            )

            assert traced_positions_of_planes[2].in_list[0][0] == tuple(
                traced_grids_of_planes[2][0]
            )
            assert traced_positions_of_planes[3].in_list[0][0] == tuple(
                traced_grids_of_planes[3][0]
            )

            assert traced_positions_of_planes[0].in_list[0][1] == tuple(
                traced_grids_of_planes[0][1]
            )

            assert traced_positions_of_planes[1].in_list[0][1] == tuple(
                traced_grids_of_planes[1][1]
            )

            assert traced_positions_of_planes[2].in_list[0][1] == tuple(
                traced_grids_of_planes[2][1]
            )
            assert traced_positions_of_planes[3].in_list[0][1] == tuple(
                traced_grids_of_planes[3][1]
            )

            assert traced_positions_of_planes[0].in_list[1][0] == tuple(
                traced_grids_of_planes[0][2]
            )

            assert traced_positions_of_planes[1].in_list[1][0] == tuple(
                traced_grids_of_planes[1][2]
            )

            assert traced_positions_of_planes[2].in_list[1][0] == tuple(
                traced_grids_of_planes[2][2]
            )
            assert traced_positions_of_planes[3].in_list[1][0] == tuple(
                traced_grids_of_planes[3][2]
            )

        def test__x2_planes__sis_lens__upper_plane_limit_removes_final_plane(
            self, sub_grid_7x7_simple, gal_x1_mp
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal_x1_mp, al.Galaxy(redshift=1.0)]
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7_simple, plane_index_limit=0
            )

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )
            assert traced_grids_of_planes[0][2] == pytest.approx(
                np.array([1.0, 1.0]), 1e-3
            )
            assert traced_grids_of_planes[0][3] == pytest.approx(
                np.array([1.0, 0.0]), 1e-3
            )

            assert len(traced_grids_of_planes) == 1

        def test__4_planes__grids_are_correct__upper_plane_limit_removes_final_planes(
            self, sub_grid_7x7_simple
        ):

            g0 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=2.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g2 = al.Galaxy(
                redshift=0.1,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g3 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g4 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g5 = al.Galaxy(
                redshift=3.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4, g5], cosmology=cosmo.Planck15
            )

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7_simple, plane_index_limit=1
            )

            # The scaling factors are as follows and were computed independently from the test_autoarray.
            beta_01 = 0.9348

            val = np.sqrt(2) / 2.0

            assert traced_grids_of_planes[0][0] == pytest.approx(
                np.array([1.0, 1.0]), 1e-4
            )
            assert traced_grids_of_planes[0][1] == pytest.approx(
                np.array([1.0, 0.0]), 1e-4
            )

            assert traced_grids_of_planes[1][0] == pytest.approx(
                np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
            )
            assert traced_grids_of_planes[1][1] == pytest.approx(
                np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
            )

            assert len(traced_grids_of_planes) == 2

    class TestProfileImages:
        def test__x1_plane__single_plane_tracer(self, sub_grid_7x7):
            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )
            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            image_plane = al.Plane(galaxies=[g0, g1, g2])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_image = image_plane.image_from_grid(grid=sub_grid_7x7)

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert tracer_image.shape_2d == (7, 7)
            assert (tracer_image == image_plane_image).all()

        def test__x2_planes__galaxy_light__no_mass__image_sum_of_image_and_source_plane(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            image_plane = al.Plane(galaxies=[g0])
            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image = image_plane.image_from_grid(
                grid=sub_grid_7x7
            ) + source_plane.image_from_grid(grid=sub_grid_7x7)

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert tracer_image.shape_2d == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x2_planes__galaxy_light_mass_sis__source_plane_image_includes_deflections(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            image_plane = al.Plane(galaxies=[g0])

            source_plane_grid = image_plane.traced_grid_from_grid(grid=sub_grid_7x7)

            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image = image_plane.image_from_grid(
                grid=sub_grid_7x7
            ) + source_plane.image_from_grid(grid=source_plane_grid)

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x2_planes__image__compare_to_galaxy_images(self, sub_grid_7x7):
            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )
            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            g0_image = g0.image_from_grid(grid=sub_grid_7x7)

            g1_image = g1.image_from_grid(grid=sub_grid_7x7)

            g2_image = g2.image_from_grid(grid=sub_grid_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert tracer_image == pytest.approx(g0_image + g1_image + g2_image, 1.0e-4)

        def test__x2_planes__returns_image_of_each_plane(self, sub_grid_7x7):
            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            g1 = al.Galaxy(
                redshift=1.0,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            image_plane = al.Plane(galaxies=[g0])

            source_plane_grid = image_plane.traced_grid_from_grid(grid=sub_grid_7x7)

            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            plane_image = image_plane.image_from_grid(
                grid=sub_grid_7x7
            ) + source_plane.image_from_grid(grid=source_plane_grid)

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert tracer_image == pytest.approx(plane_image, 1.0e-4)

        def test__x3_planes___light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(
                redshift=2.0, light_profile=al.lp.EllipticalSersic(intensity=0.3)
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0])
            plane_1 = al.Plane(galaxies=[g1])
            plane_2 = al.Plane(galaxies=[g2])

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            image = (
                plane_0.image_from_grid(grid=sub_grid_7x7)
                + plane_1.image_from_grid(grid=traced_grids_of_planes[1])
                + plane_2.image_from_grid(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert image.shape_2d == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x3_planes__galaxy_light_mass_sis__source_plane_image_includes_deflections(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(
                redshift=2.0, light_profile=al.lp.EllipticalSersic(intensity=0.3)
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            image = (
                plane_0.image_from_grid(grid=sub_grid_7x7)
                + plane_1.image_from_grid(grid=traced_grids_of_planes[1])
                + plane_2.image_from_grid(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert image.shape_2d == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x3_planes__same_as_above_more_galaxies(self, sub_grid_7x7):
            g0 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(
                redshift=2.0, light_profile=al.lp.EllipticalSersic(intensity=0.3)
            )
            g3 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.4)
            )
            g4 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.5)
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0, g3])
            plane_1 = al.Plane(galaxies=[g1, g4])
            plane_2 = al.Plane(galaxies=[g2])

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            image = (
                plane_0.image_from_grid(grid=sub_grid_7x7)
                + plane_1.image_from_grid(grid=traced_grids_of_planes[1])
                + plane_2.image_from_grid(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_from_grid(grid=sub_grid_7x7)

            assert image.shape_2d == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__images_of_planes__planes_without_light_profiles_are_all_zeros(
            self, sub_grid_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(redshift=2.0)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0])
            plane_1 = al.Plane(galaxies=[g1])

            plane_0_image = plane_0.image_from_grid(grid=sub_grid_7x7)

            plane_1_image = plane_1.image_from_grid(grid=sub_grid_7x7)

            tracer_image_of_planes = tracer.images_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert len(tracer_image_of_planes) == 3

            assert tracer_image_of_planes[0].shape_2d == (7, 7)
            assert tracer_image_of_planes[0] == pytest.approx(plane_0_image, 1.0e-4)

            assert tracer_image_of_planes[1].shape_2d == (7, 7)
            assert tracer_image_of_planes[1] == pytest.approx(plane_1_image, 1.0e-4)

            assert tracer_image_of_planes[2].shape_2d == (7, 7)
            assert (tracer_image_of_planes[2].in_2d_binned == np.zeros((7, 7))).all()

        def test__x1_plane__padded_image__compare_to_galaxy_images_using_padded_grids(
            self, sub_grid_7x7
        ):
            padded_grid = sub_grid_7x7.padded_grid_from_kernel_shape(
                kernel_shape_2d=(3, 3)
            )

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )
            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            padded_g0_image = g0.image_from_grid(grid=padded_grid)

            padded_g1_image = g1.image_from_grid(grid=padded_grid)

            padded_g2_image = g2.image_from_grid(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            padded_tracer_image = tracer.padded_image_from_grid_and_psf_shape(
                grid=sub_grid_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_2d == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

        def test__x3_planes__padded_2d_image_from_plane__mapped_correctly(
            self, sub_grid_7x7
        ):
            padded_grid = sub_grid_7x7.padded_grid_from_kernel_shape(
                kernel_shape_2d=(3, 3)
            )

            g0 = al.Galaxy(
                redshift=0.1, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(
                redshift=2.0, light_profile=al.lp.EllipticalSersic(intensity=0.3)
            )

            padded_g0_image = g0.image_from_grid(grid=padded_grid)

            padded_g1_image = g1.image_from_grid(grid=padded_grid)

            padded_g2_image = g2.image_from_grid(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            padded_tracer_image = tracer.padded_image_from_grid_and_psf_shape(
                grid=sub_grid_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_2d == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

        def test__x1_plane__padded_image__compare_to_galaxy_images_using_padded_grids_and_grid_iterato(
            self, grid_iterate_7x7
        ):
            padded_grid = grid_iterate_7x7.padded_grid_from_kernel_shape(
                kernel_shape_2d=(3, 3)
            )

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )
            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            padded_g0_image = g0.image_from_grid(grid=padded_grid)

            padded_g1_image = g1.image_from_grid(grid=padded_grid)

            padded_g2_image = g2.image_from_grid(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            padded_tracer_image = tracer.padded_image_from_grid_and_psf_shape(
                grid=grid_iterate_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_2d == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

            image = tracer.image_from_grid(grid=grid_iterate_7x7)

            assert padded_tracer_image.in_2d[4, 4] == image.in_2d[3, 3]

        def test__galaxy_image_dict_from_grid(self, sub_grid_7x7):

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=al.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            g3 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=5.0)
            )

            g0_image = g0.image_from_grid(grid=sub_grid_7x7)
            g1_image = g1.image_from_grid(grid=sub_grid_7x7)
            g2_image = g2.image_from_grid(grid=sub_grid_7x7)

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            source_grid_7x7 = sub_grid_7x7 - g1_deflections

            g3_image = g3.image_from_grid(grid=source_grid_7x7)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g3, g1, g0, g2], cosmology=cosmo.Planck15
            )

            image_1d_dict = tracer.galaxy_image_dict_from_grid(grid=sub_grid_7x7)

            assert (image_1d_dict[g0].in_1d == g0_image).all()
            assert (image_1d_dict[g1].in_1d == g1_image).all()
            assert (image_1d_dict[g2].in_1d == g2_image).all()
            assert (image_1d_dict[g3].in_1d == g3_image).all()

            image_dict = tracer.galaxy_image_dict_from_grid(grid=sub_grid_7x7)

            assert (image_dict[g0].in_2d == g0_image.in_2d).all()
            assert (image_dict[g1].in_2d == g1_image.in_2d).all()
            assert (image_dict[g2].in_2d == g2_image.in_2d).all()
            assert (image_dict[g3].in_2d == g3_image.in_2d).all()

    class TestConvergence:
        def test__galaxy_mass_sis__no_source_plane_convergence(self, sub_grid_7x7):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_convergence = image_plane.convergence_from_grid(
                grid=sub_grid_7x7
            )

            tracer_convergence = tracer.convergence_from_grid(grid=sub_grid_7x7)

            assert image_plane_convergence.shape_2d == (7, 7)
            assert (image_plane_convergence == tracer_convergence).all()

        def test__galaxy_entered_3_times__both_planes__different_convergence_for_each(
            self, sub_grid_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=3.0),
            )

            g0_convergence = g0.convergence_from_grid(grid=sub_grid_7x7)

            g1_convergence = g1.convergence_from_grid(grid=sub_grid_7x7)

            g2_convergence = g2.convergence_from_grid(grid=sub_grid_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_convergence = tracer.image_plane.convergence_from_grid(
                grid=sub_grid_7x7
            )

            source_plane_convergence = tracer.source_plane.convergence_from_grid(
                grid=sub_grid_7x7
            )

            tracer_convergence = tracer.convergence_from_grid(grid=sub_grid_7x7)

            assert image_plane_convergence == pytest.approx(
                g0_convergence + g1_convergence, 1.0e-4
            )
            assert (source_plane_convergence == g2_convergence).all()
            assert tracer_convergence == pytest.approx(
                g0_convergence + g1_convergence + g2_convergence, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(self, positions_7x7):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_convergence = g0.convergence_from_grid(grid=positions_7x7)

            g1_convergence = g1.convergence_from_grid(grid=positions_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_convergence = tracer.image_plane.convergence_from_grid(
                grid=positions_7x7
            )

            source_plane_convergence = tracer.source_plane.convergence_from_grid(
                grid=positions_7x7
            )

            tracer_convergence = tracer.convergence_from_grid(grid=positions_7x7)

            assert image_plane_convergence.in_list[0][0] == pytest.approx(
                g0_convergence.in_list[0][0] + g1_convergence.in_list[0][0], 1.0e-4
            )
            assert tracer_convergence.in_list[0][0] == pytest.approx(
                g0_convergence.in_list[0][0] + g1_convergence.in_list[0][0], 1.0e-4
            )

        def test__no_galaxy_has_mass_profile__convergence_returned_as_zeros(
            self, sub_grid_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            assert (
                tracer.convergence_from_grid(grid=sub_grid_7x7).in_2d_binned
                == np.zeros(shape=(7, 7))
            ).all()

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.1), al.Galaxy(redshift=0.2)]
            )

            assert (
                tracer.convergence_from_grid(grid=sub_grid_7x7).in_2d_binned
                == np.zeros(shape=(7, 7))
            ).all()

    class TestPotential:
        def test__galaxy_mass_sis__no_source_plane_potential(self, sub_grid_7x7):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_potential = image_plane.potential_from_grid(grid=sub_grid_7x7)

            tracer_potential = tracer.potential_from_grid(grid=sub_grid_7x7)

            assert image_plane_potential.shape_2d == (7, 7)
            assert (image_plane_potential == tracer_potential).all()

        def test__galaxy_entered_3_times__both_planes__different_potential_for_each(
            self, sub_grid_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=3.0),
            )

            g0_potential = g0.potential_from_grid(grid=sub_grid_7x7)

            g1_potential = g1.potential_from_grid(grid=sub_grid_7x7)

            g2_potential = g2.potential_from_grid(grid=sub_grid_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_potential = tracer.image_plane.potential_from_grid(
                grid=sub_grid_7x7
            )

            source_plane_potential = tracer.source_plane.potential_from_grid(
                grid=sub_grid_7x7
            )

            tracer_potential = tracer.potential_from_grid(grid=sub_grid_7x7)

            assert image_plane_potential == pytest.approx(
                g0_potential + g1_potential, 1.0e-4
            )
            assert (source_plane_potential == g2_potential).all()
            assert tracer_potential == pytest.approx(
                g0_potential + g1_potential + g2_potential, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(self, positions_7x7):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_potential = g0.potential_from_grid(grid=positions_7x7)

            g1_potential = g1.potential_from_grid(grid=positions_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_potential = tracer.image_plane.potential_from_grid(
                grid=positions_7x7
            )

            source_plane_potential = tracer.source_plane.potential_from_grid(
                grid=positions_7x7
            )

            tracer_potential = tracer.potential_from_grid(grid=positions_7x7)

            assert image_plane_potential.in_list[0][0] == pytest.approx(
                g0_potential.in_list[0][0] + g1_potential.in_list[0][0], 1.0e-4
            )
            assert tracer_potential.in_list[0][0] == pytest.approx(
                g0_potential.in_list[0][0] + g1_potential.in_list[0][0], 1.0e-4
            )

        def test__no_galaxy_has_mass_profile__potential_returned_as_zeros(
            self, sub_grid_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            assert (
                tracer.potential_from_grid(grid=sub_grid_7x7).in_2d_binned
                == np.zeros(shape=(7, 7))
            ).all()

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.1), al.Galaxy(redshift=0.2)]
            )

            assert (
                tracer.potential_from_grid(grid=sub_grid_7x7).in_2d_binned
                == np.zeros(shape=(7, 7))
            ).all()

    class TestDeflectionsOfSummedPlanes:
        def test__galaxy_mass_sis__source_plane_no_mass__deflections_is_ignored(
            self, sub_grid_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_deflections = image_plane.deflections_from_grid(
                grid=sub_grid_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from_grid(
                grid=sub_grid_7x7
            )

            assert tracer_deflections.shape_2d == (7, 7)
            assert (image_plane_deflections == tracer_deflections).all()

        def test__galaxy_entered_3_times__different_deflections_for_each(
            self, sub_grid_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )
            g2 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=3.0),
            )

            g0_deflections = g0.deflections_from_grid(grid=sub_grid_7x7)

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            g2_deflections = g2.deflections_from_grid(grid=sub_grid_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_deflections = tracer.image_plane.deflections_from_grid(
                grid=sub_grid_7x7
            )

            source_plane_deflections = tracer.source_plane.deflections_from_grid(
                grid=sub_grid_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from_grid(
                grid=sub_grid_7x7
            )

            assert image_plane_deflections == pytest.approx(
                g0_deflections + g1_deflections, 1.0e-4
            )
            assert source_plane_deflections == pytest.approx(g2_deflections, 1.0e-4)
            assert tracer_deflections == pytest.approx(
                g0_deflections + g1_deflections + g2_deflections, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(self, positions_7x7):

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_deflections = g0.deflections_from_grid(grid=positions_7x7)

            g1_deflections = g1.deflections_from_grid(grid=positions_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_deflections = tracer.image_plane.deflections_from_grid(
                grid=positions_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from_grid(
                grid=positions_7x7
            )

            assert image_plane_deflections.in_list[0][0][0] == pytest.approx(
                g0_deflections.in_list[0][0][0] + g1_deflections.in_list[0][0][0],
                1.0e-4,
            )
            assert image_plane_deflections.in_list[0][0][1] == pytest.approx(
                g0_deflections.in_list[0][0][1] + g1_deflections.in_list[0][0][1],
                1.0e-4,
            )
            assert tracer_deflections.in_list[0][0][0] == pytest.approx(
                g0_deflections.in_list[0][0][0] + g1_deflections.in_list[0][0][0],
                1.0e-4,
            )
            assert tracer_deflections.in_list[0][0][1] == pytest.approx(
                g0_deflections.in_list[0][0][1] + g1_deflections.in_list[0][0][1],
                1.0e-4,
            )

        def test__no_galaxy_has_mass_profile__deflections_returned_as_zeros(
            self, sub_grid_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from_grid(
                grid=sub_grid_7x7
            )

            assert (
                tracer_deflections.in_2d_binned[:, :, 0] == np.zeros(shape=(7, 7))
            ).all()
            assert (
                tracer_deflections.in_2d_binned[:, :, 1] == np.zeros(shape=(7, 7))
            ).all()

    class TestGridAtRedshift:
        def test__lens_z05_source_z01_redshifts__match_planes_redshifts__gives_same_grids(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=1.0
                ),
            )
            g1 = al.Galaxy(redshift=1.0)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=0.5
            )

            assert (grid_at_redshift == sub_grid_7x7).all()

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=1.0
            )

            source_plane_grid = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )[1]

            assert (grid_at_redshift == source_plane_grid).all()

        def test__same_as_above_but_for_multi_tracing(self, sub_grid_7x7):
            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=1.0
                ),
            )
            g1 = al.Galaxy(
                redshift=0.75,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=2.0
                ),
            )
            g2 = al.Galaxy(
                redshift=1.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=3.0
                ),
            )
            g3 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=4.0
                ),
            )
            g4 = al.Galaxy(redshift=2.0)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, g3, g4])

            traced_grids_of_planes = tracer.traced_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=0.5
            )

            assert grid_at_redshift == pytest.approx(traced_grids_of_planes[0], 1.0e-4)

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=0.75
            )

            assert grid_at_redshift == pytest.approx(traced_grids_of_planes[1], 1.0e-4)

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=1.0
            )

            assert grid_at_redshift == pytest.approx(traced_grids_of_planes[2], 1.0e-4)

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=1.5
            )

            assert grid_at_redshift == pytest.approx(traced_grids_of_planes[3], 1.0e-4)

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=2.0
            )

            assert grid_at_redshift == pytest.approx(traced_grids_of_planes[4], 1.0e-4)

        def test__input_redshift_between_two_planes__two_planes_between_earth_and_input_redshift(
            self, sub_grid_7x7
        ):

            sub_grid_7x7[0] = np.array([[1.0, -1.0]])
            sub_grid_7x7[1] = np.array([[1.0, 0.0]])

            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=1.0
                ),
            )
            g1 = al.Galaxy(
                redshift=0.75,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=2.0
                ),
            )
            g2 = al.Galaxy(redshift=2.0)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7, redshift=1.9
            )

            assert grid_at_redshift[0][0] == pytest.approx(-1.06587, 1.0e-1)
            assert grid_at_redshift[0][1] == pytest.approx(1.06587, 1.0e-1)
            assert grid_at_redshift[1][0] == pytest.approx(-1.921583, 1.0e-1)
            assert grid_at_redshift[1][1] == pytest.approx(0.0, 1.0e-1)

        def test__input_redshift_before_first_plane__returns_image_plane(
            self, sub_grid_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=1.0
                ),
            )
            g1 = al.Galaxy(
                redshift=0.75,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=2.0
                ),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            grid_at_redshift = tracer.grid_at_redshift_from_grid_and_redshift(
                grid=sub_grid_7x7.geometry.unmasked_grid_sub_1, redshift=0.3
            )

            assert (grid_at_redshift == sub_grid_7x7.geometry.unmasked_grid_sub_1).all()

    class TestContributionMap:
        def test__contribution_maps_are_same_as_hyper_galaxy_calculation(self):

            hyper_model_image = al.Array.manual_2d(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = al.Array.manual_2d(
                array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = al.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = al.HyperGalaxy(contribution_factor=10.0)

            galaxy_0 = al.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_1 = al.Galaxy(
                redshift=1.0,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

            assert (
                tracer.contribution_map
                == tracer.image_plane.contribution_map
                + tracer.source_plane.contribution_map
            ).all()
            assert (
                tracer.contribution_maps_of_planes[0].in_1d
                == tracer.image_plane.contribution_map
            ).all()

            assert (
                tracer.contribution_maps_of_planes[1].in_1d
                == tracer.source_plane.contribution_map
            ).all()

            galaxy_0 = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

            assert (
                tracer.contribution_map == tracer.source_plane.contribution_map
            ).all()
            assert tracer.contribution_maps_of_planes[0] == None

            assert (
                tracer.contribution_maps_of_planes[1].in_1d
                == tracer.source_plane.contribution_map
            ).all()

            galaxy_1 = al.Galaxy(redshift=1.0)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

            assert tracer.contribution_map == None
            assert tracer.contribution_maps_of_planes[0] == None

            assert tracer.contribution_maps_of_planes[1] == None

    class TestLensingObject:
        def test__correct_einstein_mass_caclulated_for_multiple_mass_profiles__means_all_innherited_methods_work(
            self,
        ):
            sis_0 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.2)

            sis_1 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.4)

            sis_2 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.6)

            sis_3 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.8)

            galaxy_0 = al.Galaxy(
                mass_profile_0=sis_0, mass_profile_1=sis_1, redshift=0.5
            )
            galaxy_1 = al.Galaxy(
                mass_profile_0=sis_2, mass_profile_1=sis_3, redshift=0.5
            )

            plane = al.Plane(galaxies=[galaxy_0, galaxy_1])

            tracer = al.Tracer(
                planes=[plane, al.Plane(redshift=1.0)], cosmology=cosmo.Planck15
            )

            assert (
                tracer.einstein_mass_angular_via_tangential_critical_curve
                == pytest.approx(np.pi * 2.0 ** 2.0, 1.0e-1)
            )


class TestAbstractTracerData:
    class TestBlurredProfileImages:
        def test__blurred_image_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
            plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

            blurred_image_0 = plane_0.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, psf=psf_3x3, blurring_grid=blurring_grid_7x7
            )

            source_grid_7x7 = plane_0.traced_grid_from_grid(grid=sub_grid_7x7)
            source_blurring_grid_7x7 = plane_0.traced_grid_from_grid(
                grid=blurring_grid_7x7
            )

            blurred_image_1 = plane_1.blurred_image_from_grid_and_psf(
                grid=source_grid_7x7,
                psf=psf_3x3,
                blurring_grid=source_blurring_grid_7x7,
            )

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

            blurred_image = tracer.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, psf=psf_3x3, blurring_grid=blurring_grid_7x7
            )

            assert blurred_image.in_1d == pytest.approx(
                blurred_image_0.in_1d + blurred_image_1.in_1d, 1.0e-4
            )

            assert blurred_image.in_2d == pytest.approx(
                blurred_image_0.in_2d + blurred_image_1.in_2d, 1.0e-4
            )

        def test__blurred_images_of_planes_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
            plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

            blurred_image_0 = plane_0.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, psf=psf_3x3, blurring_grid=blurring_grid_7x7
            )

            source_grid_7x7 = plane_0.traced_grid_from_grid(grid=sub_grid_7x7)
            source_blurring_grid_7x7 = plane_0.traced_grid_from_grid(
                grid=blurring_grid_7x7
            )

            blurred_image_1 = plane_1.blurred_image_from_grid_and_psf(
                grid=source_grid_7x7,
                psf=psf_3x3,
                blurring_grid=source_blurring_grid_7x7,
            )

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

            blurred_images = tracer.blurred_images_of_planes_from_grid_and_psf(
                grid=sub_grid_7x7, psf=psf_3x3, blurring_grid=blurring_grid_7x7
            )

            assert (blurred_images[0].in_1d == blurred_image_0.in_1d).all()
            assert (blurred_images[1].in_1d == blurred_image_1.in_1d).all()

            assert (blurred_images[0].in_2d == blurred_image_0.in_2d).all()
            assert (blurred_images[1].in_2d == blurred_image_1.in_2d).all()

        def test__blurred_image_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
            plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

            blurred_image_0 = plane_0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            source_grid_7x7 = plane_0.traced_grid_from_grid(grid=sub_grid_7x7)
            source_blurring_grid_7x7 = plane_0.traced_grid_from_grid(
                grid=blurring_grid_7x7
            )

            blurred_image_1 = plane_1.blurred_image_from_grid_and_convolver(
                grid=source_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=source_blurring_grid_7x7,
            )

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

            blurred_image = tracer.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            assert blurred_image.in_1d == pytest.approx(
                blurred_image_0.in_1d + blurred_image_1.in_1d, 1.0e-4
            )

            assert blurred_image.in_2d == pytest.approx(
                blurred_image_0.in_2d + blurred_image_1.in_2d, 1.0e-4
            )

        def test__blurred_images_of_planes_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
            plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

            blurred_image_0 = plane_0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            source_grid_7x7 = plane_0.traced_grid_from_grid(grid=sub_grid_7x7)
            source_blurring_grid_7x7 = plane_0.traced_grid_from_grid(
                grid=blurring_grid_7x7
            )

            blurred_image_1 = plane_1.blurred_image_from_grid_and_convolver(
                grid=source_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=source_blurring_grid_7x7,
            )

            tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

            blurred_images = tracer.blurred_images_of_planes_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            assert (blurred_images[0].in_1d == blurred_image_0.in_1d).all()
            assert (blurred_images[1].in_1d == blurred_image_1.in_1d).all()

            assert (blurred_images[0].in_2d == blurred_image_0.in_2d).all()
            assert (blurred_images[1].in_2d == blurred_image_1.in_2d).all()

        def test__galaxy_blurred_image_dict_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=al.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            g3 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=5.0)
            )

            g0_blurred_image = g0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            g1_blurred_image = g1.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            g2_blurred_image = g2.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            source_grid_7x7 = sub_grid_7x7 - g1_deflections

            g1_blurring_deflections = g1.deflections_from_grid(grid=blurring_grid_7x7)

            source_blurring_grid_7x7 = blurring_grid_7x7 - g1_blurring_deflections

            g3_blurred_image = g3.blurred_image_from_grid_and_convolver(
                grid=source_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=source_blurring_grid_7x7,
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g3, g1, g0, g2], cosmology=cosmo.Planck15
            )

            blurred_image_dict = (
                tracer.galaxy_blurred_image_dict_from_grid_and_convolver(
                    grid=sub_grid_7x7,
                    convolver=convolver_7x7,
                    blurring_grid=blurring_grid_7x7,
                )
            )

            assert (blurred_image_dict[g0].in_1d == g0_blurred_image.in_1d).all()
            assert (blurred_image_dict[g1].in_1d == g1_blurred_image.in_1d).all()
            assert (blurred_image_dict[g2].in_1d == g2_blurred_image.in_1d).all()
            assert (blurred_image_dict[g3].in_1d == g3_blurred_image.in_1d).all()

    class TestUnmaskedBlurredProfileImages:
        def test__unmasked_images_of_tracer_planes_and_galaxies(self):

            psf = al.Kernel.manual_2d(
                array=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
                pixel_scales=1.0,
            )

            mask = al.Mask2D.manual(
                mask=np.array(
                    [[True, True, True], [True, False, True], [True, True, True]]
                ),
                pixel_scales=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=0.2)
            )
            g2 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.3)
            )
            g3 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=0.4)
            )

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, g3])

            padded_grid = grid.padded_grid_from_kernel_shape(
                kernel_shape_2d=psf.shape_2d
            )

            traced_padded_grids = tracer.traced_grids_of_planes_from_grid(
                grid=padded_grid
            )

            manual_blurred_image_0 = tracer.image_plane.images_of_galaxies_from_grid(
                grid=traced_padded_grids[0]
            )[0]
            manual_blurred_image_0 = psf.convolved_array_from_array(
                array=manual_blurred_image_0
            )

            manual_blurred_image_1 = tracer.image_plane.images_of_galaxies_from_grid(
                grid=traced_padded_grids[0]
            )[1]
            manual_blurred_image_1 = psf.convolved_array_from_array(
                array=manual_blurred_image_1
            )

            manual_blurred_image_2 = tracer.source_plane.images_of_galaxies_from_grid(
                grid=traced_padded_grids[1]
            )[0]
            manual_blurred_image_2 = psf.convolved_array_from_array(
                array=manual_blurred_image_2
            )

            manual_blurred_image_3 = tracer.source_plane.images_of_galaxies_from_grid(
                grid=traced_padded_grids[1]
            )[1]
            manual_blurred_image_3 = psf.convolved_array_from_array(
                array=manual_blurred_image_3
            )

            unmasked_blurred_image = tracer.unmasked_blurred_image_from_grid_and_psf(
                grid=grid, psf=psf
            )

            assert unmasked_blurred_image.in_2d == pytest.approx(
                manual_blurred_image_0.in_2d_binned[1:4, 1:4]
                + manual_blurred_image_1.in_2d_binned[1:4, 1:4]
                + manual_blurred_image_2.in_2d_binned[1:4, 1:4]
                + manual_blurred_image_3.in_2d_binned[1:4, 1:4],
                1.0e-4,
            )

            unmasked_blurred_image_of_planes = (
                tracer.unmasked_blurred_image_of_planes_from_grid_and_psf(
                    grid=grid, psf=psf
                )
            )

            assert unmasked_blurred_image_of_planes[0].in_2d == pytest.approx(
                manual_blurred_image_0.in_2d_binned[1:4, 1:4]
                + manual_blurred_image_1.in_2d_binned[1:4, 1:4],
                1.0e-4,
            )
            assert unmasked_blurred_image_of_planes[1].in_2d == pytest.approx(
                manual_blurred_image_2.in_2d_binned[1:4, 1:4]
                + manual_blurred_image_3.in_2d_binned[1:4, 1:4],
                1.0e-4,
            )

            unmasked_blurred_image_of_planes_and_galaxies = (
                tracer.unmasked_blurred_image_of_planes_and_galaxies_from_grid_and_psf(
                    grid=grid, psf=psf
                )
            )

            assert (
                unmasked_blurred_image_of_planes_and_galaxies[0][0].in_2d
                == manual_blurred_image_0.in_2d_binned[1:4, 1:4]
            ).all()
            assert (
                unmasked_blurred_image_of_planes_and_galaxies[0][1].in_2d
                == manual_blurred_image_1.in_2d_binned[1:4, 1:4]
            ).all()
            assert (
                unmasked_blurred_image_of_planes_and_galaxies[1][0].in_2d
                == manual_blurred_image_2.in_2d_binned[1:4, 1:4]
            ).all()
            assert (
                unmasked_blurred_image_of_planes_and_galaxies[1][1].in_2d
                == manual_blurred_image_3.in_2d_binned[1:4, 1:4]
            ).all()

    class TestVisibilities:
        def test__visibilities_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):
            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllipticalSersic(intensity=1.0),
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            g0_image_1d = g0.image_from_grid(grid=sub_grid_7x7)

            deflections = g0.deflections_from_grid(grid=sub_grid_7x7)

            source_grid_7x7 = sub_grid_7x7 - deflections

            g1_image_1d = g1.image_from_grid(grid=source_grid_7x7)

            visibilities = transformer_7x7_7.visibilities_from_image(
                image=g0_image_1d + g1_image_1d
            )

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            tracer_visibilities = tracer.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert visibilities == pytest.approx(tracer_visibilities, 1.0e-4)

        def test__visibilities_of_planes_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=2.0)
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
            plane_1 = al.Plane(redshift=0.5, galaxies=[g1])
            plane_2 = al.Plane(redshift=1.0, galaxies=[al.Galaxy(redshift=1.0)])

            visibilities_0 = plane_0.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            visibilities_1 = plane_1.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            tracer = al.Tracer(
                planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15
            )

            visibilities = (
                tracer.profile_visibilities_of_planes_from_grid_and_transformer(
                    grid=sub_grid_7x7, transformer=transformer_7x7_7
                )
            )

            assert (visibilities[0] == visibilities_0).all()
            assert (visibilities[1] == visibilities_1).all()

        def test__galaxy_visibilities_dict_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):

            g0 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=al.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = al.Galaxy(
                redshift=0.5, light_profile=al.lp.EllipticalSersic(intensity=3.0)
            )

            g3 = al.Galaxy(
                redshift=1.0, light_profile=al.lp.EllipticalSersic(intensity=5.0)
            )

            g0_visibilities = g0.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            g1_visibilities = g1.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            g2_visibilities = g2.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            source_grid_7x7 = sub_grid_7x7 - g1_deflections

            g3_visibilities = g3.profile_visibilities_from_grid_and_transformer(
                grid=source_grid_7x7, transformer=transformer_7x7_7
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[g3, g1, g0, g2], cosmology=cosmo.Planck15
            )

            visibilities_dict = (
                tracer.galaxy_profile_visibilities_dict_from_grid_and_transformer(
                    grid=sub_grid_7x7, transformer=transformer_7x7_7
                )
            )

            assert (visibilities_dict[g0] == g0_visibilities).all()
            assert (visibilities_dict[g1] == g1_visibilities).all()
            assert (visibilities_dict[g2] == g2_visibilities).all()
            assert (visibilities_dict[g3] == g3_visibilities).all()

    class TestGridIrregularsOfPlanes:
        def test__x2_planes__traced_grid_setup_correctly(self, sub_grid_7x7):
            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(
                    value=1, grid=np.array([[1.0, 1.0]])
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            pixelization_grids = tracer.sparse_image_plane_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert pixelization_grids[0] == None
            assert (pixelization_grids[1] == np.array([[1.0, 1.0]])).all()

        def test__multi_plane__traced_grid_setup_correctly(self, sub_grid_7x7):

            galaxy_pix0 = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(
                    value=1, grid=np.array([[1.0, 1.0]])
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )

            galaxy_pix1 = al.Galaxy(
                redshift=2.0,
                pixelization=mock_inv.MockPixelization(
                    value=1, grid=np.array([[2.0, 2.0]])
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )

            galaxy_no_pix0 = al.Galaxy(redshift=0.25)
            galaxy_no_pix1 = al.Galaxy(redshift=0.5)
            galaxy_no_pix2 = al.Galaxy(redshift=1.5)

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    galaxy_pix0,
                    galaxy_pix1,
                    galaxy_no_pix0,
                    galaxy_no_pix1,
                    galaxy_no_pix2,
                ]
            )

            pixelization_grids = tracer.sparse_image_plane_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert pixelization_grids[0] == None
            assert pixelization_grids[1] == None
            assert (pixelization_grids[2] == np.array([[1.0, 1.0]])).all()
            assert pixelization_grids[3] == None
            assert (pixelization_grids[4] == np.array([[2.0, 2.0]])).all()

    class TestTracedGridIrregularsOfPlanes:
        def test__x2_planes__no_mass_profiles__traced_grid_setup_correctly(
            self, sub_grid_7x7
        ):

            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(
                    value=1,
                    grid=al.Grid.manual_2d(
                        grid=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
                    ),
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            traced_pixelization_grids = tracer.traced_sparse_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert traced_pixelization_grids[0] == None
            assert (traced_pixelization_grids[1] == np.array([[1.0, 0.0]])).all()

        def test__x2_planes__include_mass_profile__traced_grid_setup_correctly(
            self, sub_grid_7x7
        ):

            galaxy_no_pix = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=0.5
                ),
            )

            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(
                    value=1,
                    grid=al.Grid.manual_2d(
                        grid=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
                    ),
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            traced_pixelization_grids = tracer.traced_sparse_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert traced_pixelization_grids[0] == None
            assert traced_pixelization_grids[1] == pytest.approx(
                np.array([[1.0 - 0.5, 0.0]]), 1.0e-4
            )

        def test__multi_plane__traced_grid_setup_correctly(self, sub_grid_7x7):

            galaxy_pix0 = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(
                    value=1,
                    grid=al.Grid.manual_2d(
                        grid=[[[1.0, 1.0]]], pixel_scales=(1.0, 1.0)
                    ),
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )

            galaxy_pix1 = al.Galaxy(
                redshift=2.0,
                pixelization=mock_inv.MockPixelization(
                    value=1,
                    grid=al.Grid.manual_2d(
                        grid=[[[2.0, 2.0]]], pixel_scales=(1.0, 1.0)
                    ),
                ),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )

            galaxy_no_pix0 = al.Galaxy(
                redshift=0.25,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=0.5
                ),
            )
            galaxy_no_pix1 = al.Galaxy(redshift=0.5)
            galaxy_no_pix2 = al.Galaxy(redshift=1.5)

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    galaxy_pix0,
                    galaxy_pix1,
                    galaxy_no_pix0,
                    galaxy_no_pix1,
                    galaxy_no_pix2,
                ]
            )

            traced_pixelization_grids = tracer.traced_sparse_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            traced_grid_pix0 = tracer.traced_grids_of_planes_from_grid(
                grid=np.array([[1.0, 1.0]])
            )[2]
            traced_grid_pix1 = tracer.traced_grids_of_planes_from_grid(
                grid=np.array([[2.0, 2.0]])
            )[4]

            assert traced_pixelization_grids[0] == None
            assert traced_pixelization_grids[1] == None
            assert (traced_pixelization_grids[2] == traced_grid_pix0).all()
            assert traced_pixelization_grids[3] == None
            assert (traced_pixelization_grids[4] == traced_grid_pix1).all()

        def test__x2_planes__no_mass_profiles__use_real_pixelization__doesnt_crash_due_to_auto_arrays(
            self, sub_grid_7x7
        ):

            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification(shape=(3, 3)),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            traced_pixelization_grids = tracer.traced_sparse_grids_of_planes_from_grid(
                grid=sub_grid_7x7
            )

            assert traced_pixelization_grids[0] is None
            assert traced_pixelization_grids[1] is not None

    class TestMappersOfPlanes:
        def test__no_galaxy_has_pixelization__returns_list_of_nones(self, sub_grid_7x7):

            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

            mappers_of_planes = tracer.mappers_of_planes_from_grid(grid=sub_grid_7x7)
            assert mappers_of_planes == [None]

        def test__source_galaxy_has_pixelization__returns_mapper_in_list(
            self, sub_grid_7x7
        ):
            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            mapper_of_planes = tracer.mappers_of_planes_from_grid(grid=sub_grid_7x7)

            assert mapper_of_planes == [None, 1]

        def test__multiplane__correct_galaxy_planes_galaxies_have_pixelization__returns_both_mappers(
            self, sub_grid_7x7
        ):

            galaxy_no_pix0 = al.Galaxy(
                redshift=0.25,
                mass_profile=al.mp.SphericalIsothermal(
                    centre=(0.0, 0.0), einstein_radius=0.5
                ),
            )
            galaxy_no_pix1 = al.Galaxy(redshift=0.5)
            galaxy_no_pix2 = al.Galaxy(redshift=1.5)

            galaxy_pix_0 = al.Galaxy(
                redshift=0.75,
                pixelization=mock_inv.MockPixelization(value=1),
                regularization=mock_inv.MockRegularization(matrix_shape=(3, 3)),
            )
            galaxy_pix_1 = al.Galaxy(
                redshift=2.0,
                pixelization=mock_inv.MockPixelization(value=2),
                regularization=mock_inv.MockRegularization(matrix_shape=(4, 4)),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[
                    galaxy_no_pix0,
                    galaxy_no_pix1,
                    galaxy_no_pix2,
                    galaxy_pix_0,
                    galaxy_pix_1,
                ]
            )

            mappers_of_planes = tracer.mappers_of_planes_from_grid(grid=sub_grid_7x7)

            assert mappers_of_planes == [None, None, 1, None, 2]

    class TestInversion:
        def test__x1_inversion_imaging_in_tracer__performs_inversion_correctly(
            self, sub_grid_7x7, masked_imaging_7x7
        ):

            pix = al.pix.Rectangular(shape=(3, 3))
            reg = al.reg.Constant(coefficient=0.0)

            g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

            tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

            inversion = tracer.inversion_imaging_from_grid_and_data(
                grid=sub_grid_7x7,
                image=masked_imaging_7x7.image,
                noise_map=masked_imaging_7x7.noise_map,
                convolver=masked_imaging_7x7.convolver,
                settings_pixelization=al.SettingsPixelization(use_border=False),
            )

            assert inversion.mapped_reconstructed_image == pytest.approx(
                masked_imaging_7x7.image, 1.0e-2
            )

        def test__x1_inversion_interferometer_in_tracer__performs_inversion_correctly(
            self, sub_grid_7x7, masked_interferometer_7
        ):

            masked_interferometer_7.visibilities = al.Visibilities.ones(shape_1d=(7,))

            pix = al.pix.Rectangular(shape=(7, 7))
            reg = al.reg.Constant(coefficient=0.0)

            g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

            tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

            inversion = tracer.inversion_interferometer_from_grid_and_data(
                grid=sub_grid_7x7,
                visibilities=masked_interferometer_7.visibilities,
                noise_map=masked_interferometer_7.noise_map,
                transformer=masked_interferometer_7.transformer,
                settings_pixelization=al.SettingsPixelization(use_border=False),
            )

            # assert inversion.mapped_reconstructed_visibilities[:, 0] == pytest.approx(
            #     masked_interferometer_7.visibilities[:, 0], 1.0e-2
            # )

    class TestHyperNoiseMap:
        def test__hyper_noise_maps_of_planes(self, sub_grid_7x7):

            noise_map_1d = al.Array.manual_2d(array=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

            hyper_model_image = al.Array.manual_2d(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = al.Array.manual_2d(
                array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = al.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = al.HyperGalaxy(contribution_factor=10.0)

            galaxy_0 = al.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
                hyper_minimum_value=0.0,
            )

            galaxy_1 = al.Galaxy(
                redshift=1.0,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
                hyper_minimum_value=0.0,
            )

            plane_0 = al.Plane(redshift=0.5, galaxies=[galaxy_0])
            plane_1 = al.Plane(redshift=0.5, galaxies=[galaxy_1])
            plane_2 = al.Plane(redshift=1.0, galaxies=[al.Galaxy(redshift=0.5)])

            hyper_noise_map_0 = plane_0.hyper_noise_map_from_noise_map(
                noise_map=noise_map_1d
            )
            hyper_noise_map_1 = plane_1.hyper_noise_map_from_noise_map(
                noise_map=noise_map_1d
            )

            tracer = al.Tracer(
                planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15
            )

            hyper_noise_maps = tracer.hyper_noise_maps_of_planes_from_noise_map(
                noise_map=noise_map_1d
            )

            assert (hyper_noise_maps[0].in_1d == hyper_noise_map_0).all()
            assert (hyper_noise_maps[1].in_1d == hyper_noise_map_1).all()
            assert hyper_noise_maps[2].in_1d == np.zeros(shape=(3, 1))

            hyper_noise_map = tracer.hyper_noise_map_from_noise_map(
                noise_map=noise_map_1d
            )

            assert (
                hyper_noise_map.in_1d == hyper_noise_map_0 + hyper_noise_map_1
            ).all()

            tracer = al.Tracer.from_galaxies(
                galaxies=[galaxy_0, galaxy_1], cosmology=cosmo.Planck15
            )

            hyper_noise_maps = tracer.hyper_noise_maps_of_planes_from_noise_map(
                noise_map=noise_map_1d
            )

            assert (hyper_noise_maps[0].in_1d == hyper_noise_map_0).all()
            assert (hyper_noise_maps[1].in_1d == hyper_noise_map_1).all()


class TestTracer:
    def test__x2_planes__no_galaxy__all_deflections_are_zeros(
        self, sub_grid_7x7_simple
    ):

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        traced_deflections_between_planes = tracer.deflections_between_planes_from_grid(
            grid=sub_grid_7x7_simple, plane_i=0, plane_j=0
        )

        assert traced_deflections_between_planes[0] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[1] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[2] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[3] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )

        traced_deflections_between_planes = tracer.deflections_between_planes_from_grid(
            grid=sub_grid_7x7_simple, plane_i=0, plane_j=1
        )

        assert traced_deflections_between_planes[0] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[1] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[2] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[3] == pytest.approx(
            np.array([0.0, 0.0]), 1e-3
        )

    def test__x2_planes__sis_lens__traced_deflection_are_correct(
        self, sub_grid_7x7_simple, gal_x1_mp
    ):

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_mp, al.Galaxy(redshift=1.0)])

        traced_deflections_between_planes = tracer.deflections_between_planes_from_grid(
            grid=sub_grid_7x7_simple, plane_i=0, plane_j=1
        )

        assert traced_deflections_between_planes[0] == pytest.approx(
            np.array([0.707, 0.707]), 1e-3
        )
        assert traced_deflections_between_planes[1] == pytest.approx(
            np.array([1.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[2] == pytest.approx(
            np.array([0.707, 0.707]), 1e-3
        )
        assert traced_deflections_between_planes[3] == pytest.approx(
            np.array([1.0, 0.0]), 1e-3
        )

    def test__same_as_above_but_x2_sis_lenses__deflections_double(
        self, sub_grid_7x7_simple, gal_x1_mp
    ):

        tracer = al.Tracer.from_galaxies(
            galaxies=[gal_x1_mp, gal_x1_mp, al.Galaxy(redshift=1.0)]
        )

        traced_deflections_between_planes = tracer.deflections_between_planes_from_grid(
            grid=sub_grid_7x7_simple, plane_i=0, plane_j=1
        )

        assert traced_deflections_between_planes[0] == pytest.approx(
            np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3
        )
        assert traced_deflections_between_planes[1] == pytest.approx(
            np.array([2.0 * 1.0, 0.0]), 1e-3
        )
        assert traced_deflections_between_planes[2] == pytest.approx(
            np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3
        )
        assert traced_deflections_between_planes[3] == pytest.approx(
            np.array([2.0 * 1.0, 0.0]), 1e-3
        )


class TestTacerFixedSlices:
    def test__6_galaxies__tracer_planes_are_correct(self, sub_grid_7x7):
        lens_g0 = al.Galaxy(redshift=0.5)
        source_g0 = al.Galaxy(redshift=2.0)
        los_g0 = al.Galaxy(redshift=0.1)
        los_g1 = al.Galaxy(redshift=0.2)
        los_g2 = al.Galaxy(redshift=0.4)
        los_g3 = al.Galaxy(redshift=0.6)

        tracer = al.Tracer.sliced_tracer_from_lens_line_of_sight_and_source_galaxies(
            lens_galaxies=[lens_g0],
            line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
            source_galaxies=[source_g0],
            planes_between_lenses=[1, 1],
            cosmology=cosmo.Planck15,
        )

        # Plane redshifts are [0.25, 0.5, 1.25, 2.0]

        assert tracer.planes[0].galaxies == [los_g0, los_g1]
        assert tracer.planes[1].galaxies == [lens_g0, los_g2, los_g3]
        assert tracer.planes[2].galaxies == []
        assert tracer.planes[3].galaxies == [source_g0]

    def test__4_planes__data_grid_and_deflections_stacks_are_correct__sis_mass_profile(
        self, sub_grid_7x7_simple
    ):

        lens_g0 = al.Galaxy(
            redshift=0.5, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        source_g0 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        los_g0 = al.Galaxy(
            redshift=0.1, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        los_g1 = al.Galaxy(
            redshift=0.2, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        los_g2 = al.Galaxy(
            redshift=0.4, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        los_g3 = al.Galaxy(
            redshift=0.6, mass_profile=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )

        tracer = al.Tracer.sliced_tracer_from_lens_line_of_sight_and_source_galaxies(
            lens_galaxies=[lens_g0],
            line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
            source_galaxies=[source_g0],
            planes_between_lenses=[1, 1],
            cosmology=cosmo.Planck15,
        )

        traced_grids = tracer.traced_grids_of_planes_from_grid(grid=sub_grid_7x7_simple)

        # This test_autoarray is essentially the same as the TracerMulti test_autoarray, we just slightly change how many galaxies go
        # in each plane and therefore change the factor in front of val for different planes.

        # The scaling factors are as follows and were computed indepedently from the test_autoarray.
        beta_01 = 0.57874474423
        beta_02 = 0.91814281
        # Beta_03 = 1.0
        beta_12 = 0.8056827034
        # Beta_13 = 1.0
        # Beta_23 = 1.0

        val = np.sqrt(2) / 2.0

        assert traced_grids[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
        assert traced_grids[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

        assert traced_grids[1][0] == pytest.approx(
            np.array([(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]), 1e-4
        )
        assert traced_grids[1][1] == pytest.approx(
            np.array([(1.0 - beta_01 * 2.0), 0.0]), 1e-4
        )

        #  Galaxies in this plane, so multiply by 3

        defl11 = 3.0 * lens_g0.deflections_from_grid(
            grid=np.array([[(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]])
        )
        defl12 = 3.0 * lens_g0.deflections_from_grid(
            grid=np.array([[(1.0 - beta_01 * 2.0 * 1.0), 0.0]])
        )

        assert traced_grids[2][0] == pytest.approx(
            np.array(
                [
                    (1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 0]),
                    (1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 1]),
                ]
            ),
            1e-4,
        )
        assert traced_grids[2][1] == pytest.approx(
            np.array([(1.0 - beta_02 * 2.0 - beta_12 * defl12[0, 0]), 0.0]), 1e-4
        )

        assert traced_grids[3][0] == pytest.approx(np.array([-2.5355, -2.5355]), 1e-4)
        assert traced_grids[3][1] == pytest.approx(np.array([2.0, 0.0]), 1e-4)


class TestRegression:
    def test__centre_of_profile_in_right_place(self):
        grid = al.Grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = tracer.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] > 0
        assert deflections.in_2d[2, 4, 0] < 0
        assert deflections.in_2d[1, 4, 1] > 0
        assert deflections.in_2d[1, 3, 1] < 0

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = tracer.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] > 0
        assert deflections.in_2d[2, 4, 0] < 0
        assert deflections.in_2d[1, 4, 1] > 0
        assert deflections.in_2d[1, 3, 1] < 0

        grid = al.GridIterate.uniform(
            shape_2d=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = tracer.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= 0
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = tracer.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= -1e-8
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0


class TestDecorators:
    def test__grid_iterate_in__iterates_array_result_correctly(self, gal_x1_lp):

        mask = al.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_lp])

        image = tracer.image_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid.from_mask(mask=mask_sub_2)
        image_sub_2 = tracer.image_from_grid(grid=grid_sub_2).in_1d_binned

        assert (image == image_sub_2).all()

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy])

        image = tracer.image_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = al.Grid.from_mask(mask=mask_sub_4)
        image_sub_4 = tracer.image_from_grid(grid=grid_sub_4).in_1d_binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid.from_mask(mask=mask_sub_8)
        image_sub_8 = tracer.image_from_grid(grid=grid_sub_8).in_1d_binned

        assert image[4] == image_sub_8[4]

    def test__grid_iterate_in__method_returns_array_list__uses_highest_sub_size_of_iterate(
        self, gal_x1_lp
    ):

        mask = al.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_lp])

        images = tracer.images_of_planes_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid.from_mask(mask=mask_sub_2)
        image_sub_2 = tracer.image_from_grid(grid=grid_sub_2).in_1d_binned

        assert (images[0] == image_sub_2).all()

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy])

        images = tracer.images_of_planes_from_grid(grid=grid)

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid.from_mask(mask=mask_sub_8)
        image_sub_8 = tracer.image_from_grid(grid=grid_sub_8).in_1d_binned

        assert images[0][4] == image_sub_8[4]

    def test__grid_iterate_in__iterates_grid_result_correctly(self, gal_x1_mp):

        mask = al.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        deflections = tracer.deflections_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid.from_mask(mask=mask_sub_2)
        deflections_sub_2 = tracer.deflections_from_grid(grid=grid_sub_2).in_1d_binned

        assert (deflections == deflections_sub_2).all()

        grid = al.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        deflections = tracer.deflections_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = al.Grid.from_mask(mask=mask_sub_4)
        deflections_sub_4 = tracer.deflections_from_grid(grid=grid_sub_4).in_1d_binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid.from_mask(mask=mask_sub_8)
        deflections_sub_8 = galaxy.deflections_from_grid(grid=grid_sub_8).in_1d_binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]

    def test__grid_interp_in__interps_based_on_intepolate_config(self):
        # `False` in interpolate.ini

        mask = al.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_interp = al.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        light_profile = al.lp.EllipticalSersic(intensity=1.0)
        light_profile_interp = al.lp.SphericalSersic(intensity=1.0)

        image_no_interp = light_profile.image_from_grid(grid=grid)

        array_interp = light_profile.image_from_grid(grid=grid_interp.grid_interp)
        image_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = al.Galaxy(
            redshift=0.5, light=light_profile_interp, light_0=light_profile
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        image = tracer.image_from_grid(grid=grid_interp)

        assert (image == image_no_interp + image_interp).all()

        mass_profile = al.mp.EllipticalIsothermal(einstein_radius=1.0)
        mass_profile_interp = al.mp.SphericalIsothermal(
            einstein_radius=3.0, centre=(0.1, 0.1)
        )

        convergence_no_interp = mass_profile.convergence_from_grid(grid=grid)

        array_interp = mass_profile_interp.convergence_from_grid(
            grid=grid_interp.grid_interp
        )
        convergence_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = al.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_from_grid(grid=grid_interp)

        assert (convergence == convergence_no_interp + convergence_interp).all()

        potential_no_interp = mass_profile.potential_from_grid(grid=grid)

        array_interp = mass_profile_interp.potential_from_grid(
            grid=grid_interp.grid_interp
        )
        potential_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = al.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        potential = tracer.potential_from_grid(grid=grid_interp)

        assert (potential == potential_no_interp + potential_interp).all()

        deflections_no_interp = mass_profile.deflections_from_grid(grid=grid)

        grid_interp_0 = mass_profile_interp.deflections_from_grid(
            grid=grid_interp.grid_interp
        )
        deflections_interp = grid_interp.interpolated_grid_from_grid_interp(
            grid_interp=grid_interp_0
        )

        galaxy = al.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        deflections = tracer.deflections_from_grid(grid=grid_interp)

        assert isinstance(deflections, al.Grid)
        assert (deflections == deflections_no_interp + deflections_interp).all()

        traced_grids = tracer.traced_grids_of_planes_from_grid(grid=grid_interp)

        assert isinstance(traced_grids[0], al.GridInterpolate)
        assert (traced_grids[0] == grid_interp).all()
        source_plane_grid = traced_grids[0] - deflections
        assert (traced_grids[1] == source_plane_grid).all()

        grid = al.Grid.from_mask(mask=mask)
        traced_grids_no_interp = tracer.traced_grids_of_planes_from_grid(grid=grid)
        assert (traced_grids[1][0, 0] != traced_grids_no_interp[1][0, 0]).all()
