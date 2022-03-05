from astropy import cosmology as cosmo
import numpy as np
import pytest
import os
from os import path
import shutil
from skimage import measure

import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def critical_curve_via_magnification_via_tracer_from(tracer, grid):
    magnification = tracer.magnification_2d_from(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.native, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.mask.grid_scaled_for_marching_squares_from(
            grid_pixels_1d=pixel_coord, shape_native=magnification.sub_shape_native
        )

        critical_curve = np.array(grid=critical_curve)

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_via_tracer_from(tracer, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_via_tracer_from(
        tracer=tracer, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = tracer.deflections_yx_2d_from(grid=critical_curve)

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
            gal_mp = al.Galaxy(redshift=0.5, mass_profile=al.mp.SphIsothermal())

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

        def test_plane_with_galaxy(self, sub_grid_2d_7x7):

            g1 = al.Galaxy(redshift=1)
            g2 = al.Galaxy(redshift=2)

            tracer = al.Tracer.from_galaxies(galaxies=[g1, g2])

            assert tracer.plane_with_galaxy(g1).galaxies == [g1]
            assert tracer.plane_with_galaxy(g2).galaxies == [g2]

        def test__has_galaxy_with_mass_profile(self, sub_grid_2d_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_mp = al.Galaxy(redshift=0.5, mass_profile=al.mp.SphIsothermal())

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
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.plane_indexes_with_pixelizations == []

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.plane_indexes_with_pixelizations == [0]

            gal_pix = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.plane_indexes_with_pixelizations == [1]

            gal_pix_0 = al.Galaxy(
                redshift=0.6,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
            )

            gal_pix_1 = al.Galaxy(
                redshift=2.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
            )

            gal0 = al.Galaxy(redshift=0.25)
            gal1 = al.Galaxy(redshift=0.5)
            gal2 = al.Galaxy(redshift=0.75)

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal_pix_0, gal_pix_1, gal0, gal1, gal2]
            )

            assert tracer.plane_indexes_with_pixelizations == [2, 4]

        def test__has_galaxy_with_pixelization(self, sub_grid_2d_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
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

        def test__has_galaxy_with_regularization(self, sub_grid_2d_7x7):
            gal = al.Galaxy(redshift=0.5)
            gal_lp = al.Galaxy(redshift=0.5, light_profile=al.lp.LightProfile())
            gal_reg = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
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

        def test__has_galaxy_with_hyper_galaxy(self, sub_grid_2d_7x7):

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

        def test__hyper_galaxy_image_list_of_planes(self, sub_grid_2d_7x7):

            gal = al.Galaxy(redshift=0.5)
            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

            assert tracer.hyper_galaxy_image_pg_list == [[]]

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal_pix])

            assert tracer.hyper_galaxy_image_pg_list == [[None, None]]

            gal_pix = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
                hyper_galaxy_image=1,
            )

            tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

            assert tracer.hyper_galaxy_image_pg_list == [[1]]

            gal0 = al.Galaxy(redshift=0.25)
            gal1 = al.Galaxy(redshift=0.75)
            gal2 = al.Galaxy(redshift=1.5)

            gal_pix0 = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
                hyper_galaxy_image=1,
            )

            gal_pix1 = al.Galaxy(
                redshift=2.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
                hyper_galaxy_image=2,
            )

            gal_pix2 = al.Galaxy(
                redshift=2.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(),
                hyper_galaxy_image=3,
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[gal0, gal1, gal2, gal_pix0, gal_pix1, gal_pix2]
            )

            assert tracer.hyper_galaxy_image_pg_list == [[], [1], [], [], [2, 3]]

    class TestPixelizations:
        def test__pixelization_list_of_lists(self, sub_grid_2d_7x7):
            galaxy_pix = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(mapper=1),
                regularization=al.m.MockRegularization(),
            )
            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

            assert tracer.pixelization_pg_list[0] == []
            assert tracer.pixelization_pg_list[1][0].mapper == 1

            galaxy_pix_0 = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(mapper=1),
                regularization=al.m.MockRegularization(),
            )

            galaxy_pix_1 = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(mapper=2),
                regularization=al.m.MockRegularization(),
            )

            galaxy_pix_2 = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(mapper=3),
                regularization=al.m.MockRegularization(),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[galaxy_pix_0, galaxy_pix_1, galaxy_pix_2]
            )

            assert tracer.pixelization_pg_list[0][0].mapper == 1
            assert tracer.pixelization_pg_list[1][0].mapper == 2
            assert tracer.pixelization_pg_list[1][1].mapper == 3

            galaxy_no_pix = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

            assert tracer.pixelization_pg_list == [[]]

        def test__regularization_list_of_lists(self, sub_grid_2d_7x7):

            galaxy_reg = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(regularization_matrix=1),
            )
            galaxy_no_reg = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_reg])

            assert tracer.regularization_pg_list[0] == []
            assert tracer.regularization_pg_list[1][0].regularization_matrix == 1
            assert tracer.regularization_list[0].regularization_matrix == 1

            galaxy_reg_0 = al.Galaxy(
                redshift=0.5,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(regularization_matrix=1),
            )

            galaxy_reg_1 = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(regularization_matrix=2),
            )

            galaxy_reg_2 = al.Galaxy(
                redshift=1.0,
                pixelization=al.m.MockPixelization(),
                regularization=al.m.MockRegularization(regularization_matrix=3),
            )

            tracer = al.Tracer.from_galaxies(
                galaxies=[galaxy_reg_0, galaxy_reg_1, galaxy_reg_2]
            )

            assert tracer.regularization_pg_list[0][0].regularization_matrix == 1
            assert tracer.regularization_pg_list[1][0].regularization_matrix == 2
            assert tracer.regularization_pg_list[1][1].regularization_matrix == 3
            assert tracer.regularization_list[0].regularization_matrix == 1
            assert tracer.regularization_list[1].regularization_matrix == 2
            assert tracer.regularization_list[2].regularization_matrix == 3

            galaxy_no_reg = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_no_reg])

            assert tracer.regularization_pg_list == [[]]

    class TestGalaxyLists:
        def test__galaxies__comes_in_plane_redshift_order(self, sub_grid_2d_7x7):
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


class TestAbstractTracerLensing:
    class TestTracedGridsFromGrid:
        def test__traced_grid_list_from__same_as_util(self, sub_grid_2d_7x7_simple):

            g0 = al.Galaxy(
                redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g2 = al.Galaxy(
                redshift=0.1, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g3 = al.Galaxy(
                redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g4 = al.Galaxy(
                redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g5 = al.Galaxy(
                redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )

            galaxies = [g0, g1, g2, g3, g4, g5]

            tracer = al.Tracer.from_galaxies(
                galaxies=galaxies, cosmology=cosmo.Planck15
            )

            traced_grid_list = tracer.traced_grid_list_from(
                grid=sub_grid_2d_7x7_simple, plane_index_limit=1
            )

            planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

            traced_grid_list_via_util = al.util.ray_tracing.traced_grid_list_from(
                planes=planes, grid=sub_grid_2d_7x7_simple, plane_index_limit=1
            )

            assert traced_grid_list[0] == pytest.approx(
                traced_grid_list_via_util[0], 1.0e-4
            )
            assert traced_grid_list[1] == pytest.approx(
                traced_grid_list_via_util[1], 1.0e-4
            )
            assert len(traced_grid_list) == 2

    class TestProfileImages:
        def test__x1_plane__single_plane_tracer(self, sub_grid_2d_7x7):
            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
            g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=3.0))

            image_plane = al.Plane(galaxies=[g0, g1, g2])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_image = image_plane.image_2d_from(grid=sub_grid_2d_7x7)

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert tracer_image.shape_native == (7, 7)
            assert (tracer_image == image_plane_image).all()

        def test__x2_planes__galaxy_light__no_mass__image_sum_of_image_and_source_plane(
            self, sub_grid_2d_7x7
        ):
            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=2.0))

            image_plane = al.Plane(galaxies=[g0])
            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image = image_plane.image_2d_from(
                grid=sub_grid_2d_7x7
            ) + source_plane.image_2d_from(grid=sub_grid_2d_7x7)

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert tracer_image.shape_native == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x2_planes__galaxy_light_mass_sis__source_plane_image_includes_deflections(
            self, sub_grid_2d_7x7
        ):
            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllSersic(intensity=1.0),
                mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
            )
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=2.0))

            image_plane = al.Plane(galaxies=[g0])

            source_plane_grid = image_plane.traced_grid_from(grid=sub_grid_2d_7x7)

            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image = image_plane.image_2d_from(
                grid=sub_grid_2d_7x7
            ) + source_plane.image_2d_from(grid=source_plane_grid)

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x2_planes__image__compare_to_galaxy_images(self, sub_grid_2d_7x7):
            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
            g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=3.0))

            g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)

            g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)

            g2_image = g2.image_2d_from(grid=sub_grid_2d_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert tracer_image == pytest.approx(g0_image + g1_image + g2_image, 1.0e-4)

        def test__x2_planes__returns_image_of_each_plane(self, sub_grid_2d_7x7):
            g0 = al.Galaxy(
                redshift=0.5,
                light_profile=al.lp.EllSersic(intensity=1.0),
                mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
            )

            g1 = al.Galaxy(
                redshift=1.0,
                light_profile=al.lp.EllSersic(intensity=1.0),
                mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
            )

            image_plane = al.Plane(galaxies=[g0])

            source_plane_grid = image_plane.traced_grid_from(grid=sub_grid_2d_7x7)

            source_plane = al.Plane(galaxies=[g1])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            plane_image = image_plane.image_2d_from(
                grid=sub_grid_2d_7x7
            ) + source_plane.image_2d_from(grid=source_plane_grid)

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert tracer_image == pytest.approx(plane_image, 1.0e-4)

        def test__x3_planes___light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(
            self, sub_grid_2d_7x7
        ):
            g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.1))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.2))
            g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.EllSersic(intensity=0.3))

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0])
            plane_1 = al.Plane(galaxies=[g1])
            plane_2 = al.Plane(galaxies=[g2])

            traced_grids_of_planes = tracer.traced_grid_list_from(grid=sub_grid_2d_7x7)

            image = (
                plane_0.image_2d_from(grid=sub_grid_2d_7x7)
                + plane_1.image_2d_from(grid=traced_grids_of_planes[1])
                + plane_2.image_2d_from(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert image.shape_native == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x3_planes__galaxy_light_mass_sis__source_plane_image_includes_deflections(
            self, sub_grid_2d_7x7
        ):
            g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.1))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.2))
            g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.EllSersic(intensity=0.3))

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            traced_grids_of_planes = tracer.traced_grid_list_from(grid=sub_grid_2d_7x7)

            image = (
                plane_0.image_2d_from(grid=sub_grid_2d_7x7)
                + plane_1.image_2d_from(grid=traced_grids_of_planes[1])
                + plane_2.image_2d_from(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert image.shape_native == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__x3_planes__same_as_above_more_galaxies(self, sub_grid_2d_7x7):
            g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.1))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.2))
            g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.EllSersic(intensity=0.3))
            g3 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.4))
            g4 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.5))

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2, g3, g4], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0, g3])
            plane_1 = al.Plane(galaxies=[g1, g4])
            plane_2 = al.Plane(galaxies=[g2])

            traced_grids_of_planes = tracer.traced_grid_list_from(grid=sub_grid_2d_7x7)

            image = (
                plane_0.image_2d_from(grid=sub_grid_2d_7x7)
                + plane_1.image_2d_from(grid=traced_grids_of_planes[1])
                + plane_2.image_2d_from(grid=traced_grids_of_planes[2])
            )

            tracer_image = tracer.image_2d_from(grid=sub_grid_2d_7x7)

            assert image.shape_native == (7, 7)
            assert image == pytest.approx(tracer_image, 1.0e-4)

        def test__images_of_planes__planes_without_light_profiles_are_all_zeros(
            self, sub_grid_2d_7x7
        ):

            g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.1))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.2))
            g2 = al.Galaxy(redshift=2.0)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            plane_0 = al.Plane(galaxies=[g0])
            plane_1 = al.Plane(galaxies=[g1])

            plane_0_image = plane_0.image_2d_from(grid=sub_grid_2d_7x7)

            plane_1_image = plane_1.image_2d_from(grid=sub_grid_2d_7x7)

            tracer_image_of_planes = tracer.image_2d_list_from(grid=sub_grid_2d_7x7)

            assert len(tracer_image_of_planes) == 3

            assert tracer_image_of_planes[0].shape_native == (7, 7)
            assert tracer_image_of_planes[0] == pytest.approx(plane_0_image, 1.0e-4)

            assert tracer_image_of_planes[1].shape_native == (7, 7)
            assert tracer_image_of_planes[1] == pytest.approx(plane_1_image, 1.0e-4)

            assert tracer_image_of_planes[2].shape_native == (7, 7)
            assert (tracer_image_of_planes[2].binned.native == np.zeros((7, 7))).all()

        def test__x1_plane__padded_image__compare_to_galaxy_images_using_padded_grids(
            self, sub_grid_2d_7x7
        ):
            padded_grid = sub_grid_2d_7x7.padded_grid_from(kernel_shape_native=(3, 3))

            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
            g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=3.0))

            padded_g0_image = g0.image_2d_from(grid=padded_grid)

            padded_g1_image = g1.image_2d_from(grid=padded_grid)

            padded_g2_image = g2.image_2d_from(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            padded_tracer_image = tracer.padded_image_2d_from(
                grid=sub_grid_2d_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_native == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

        def test__x3_planes__padded_2d_image_from_plane__mapped_correctly(
            self, sub_grid_2d_7x7
        ):
            padded_grid = sub_grid_2d_7x7.padded_grid_from(kernel_shape_native=(3, 3))

            g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.EllSersic(intensity=0.1))
            g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=0.2))
            g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.EllSersic(intensity=0.3))

            padded_g0_image = g0.image_2d_from(grid=padded_grid)

            padded_g1_image = g1.image_2d_from(grid=padded_grid)

            padded_g2_image = g2.image_2d_from(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g0, g1, g2], cosmology=cosmo.Planck15
            )

            padded_tracer_image = tracer.padded_image_2d_from(
                grid=sub_grid_2d_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_native == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

        def test__x1_plane__padded_image__compare_to_galaxy_images_using_padded_grids_and_grid_iterato(
            self, grid_2d_iterate_7x7
        ):
            padded_grid = grid_2d_iterate_7x7.padded_grid_from(
                kernel_shape_native=(3, 3)
            )

            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
            g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=3.0))

            padded_g0_image = g0.image_2d_from(grid=padded_grid)

            padded_g1_image = g1.image_2d_from(grid=padded_grid)

            padded_g2_image = g2.image_2d_from(grid=padded_grid)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            padded_tracer_image = tracer.padded_image_2d_from(
                grid=grid_2d_iterate_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_tracer_image.shape_native == (9, 9)
            assert padded_tracer_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

            image = tracer.image_2d_from(grid=grid_2d_iterate_7x7)

            assert padded_tracer_image.native[4, 4] == image.native[3, 3]

        def test__galaxy_image_2d_dict_from(self, sub_grid_2d_7x7):

            g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
            g1 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
                light_profile=al.lp.EllSersic(intensity=2.0),
            )

            g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=3.0))

            g3 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=5.0))

            g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)
            g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)
            g2_image = g2.image_2d_from(grid=sub_grid_2d_7x7)

            g1_deflections = g1.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

            source_grid_2d_7x7 = sub_grid_2d_7x7 - g1_deflections

            g3_image = g3.image_2d_from(grid=source_grid_2d_7x7)

            tracer = al.Tracer.from_galaxies(
                galaxies=[g3, g1, g0, g2], cosmology=cosmo.Planck15
            )

            image_1d_dict = tracer.galaxy_image_2d_dict_from(grid=sub_grid_2d_7x7)

            assert (image_1d_dict[g0].slim == g0_image).all()
            assert (image_1d_dict[g1].slim == g1_image).all()
            assert (image_1d_dict[g2].slim == g2_image).all()
            assert (image_1d_dict[g3].slim == g3_image).all()

            image_dict = tracer.galaxy_image_2d_dict_from(grid=sub_grid_2d_7x7)

            assert (image_dict[g0].native == g0_image.native).all()
            assert (image_dict[g1].native == g1_image.native).all()
            assert (image_dict[g2].native == g2_image.native).all()
            assert (image_dict[g3].native == g3_image.native).all()

    class TestConvergence:
        def test__galaxy_mass_sis__no_source_plane_convergence(self, sub_grid_2d_7x7):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_convergence = image_plane.convergence_2d_from(
                grid=sub_grid_2d_7x7
            )

            tracer_convergence = tracer.convergence_2d_from(grid=sub_grid_2d_7x7)

            assert image_plane_convergence.shape_native == (7, 7)
            assert (image_plane_convergence == tracer_convergence).all()

        def test__galaxy_entered_3_times__both_planes__different_convergence_for_each(
            self, sub_grid_2d_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )
            g2 = al.Galaxy(
                redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=3.0)
            )

            g0_convergence = g0.convergence_2d_from(grid=sub_grid_2d_7x7)

            g1_convergence = g1.convergence_2d_from(grid=sub_grid_2d_7x7)

            g2_convergence = g2.convergence_2d_from(grid=sub_grid_2d_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_convergence = tracer.image_plane.convergence_2d_from(
                grid=sub_grid_2d_7x7
            )

            source_plane_convergence = tracer.source_plane.convergence_2d_from(
                grid=sub_grid_2d_7x7
            )

            tracer_convergence = tracer.convergence_2d_from(grid=sub_grid_2d_7x7)

            assert image_plane_convergence == pytest.approx(
                g0_convergence + g1_convergence, 1.0e-4
            )
            assert (source_plane_convergence == g2_convergence).all()
            assert tracer_convergence == pytest.approx(
                g0_convergence + g1_convergence + g2_convergence, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(
            self, grid_2d_irregular_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )

            g0_convergence = g0.convergence_2d_from(grid=grid_2d_irregular_7x7)

            g1_convergence = g1.convergence_2d_from(grid=grid_2d_irregular_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_convergence = tracer.image_plane.convergence_2d_from(
                grid=grid_2d_irregular_7x7
            )

            tracer_convergence = tracer.convergence_2d_from(grid=grid_2d_irregular_7x7)

            assert image_plane_convergence.in_list[0] == pytest.approx(
                g0_convergence.in_list[0] + g1_convergence.in_list[0], 1.0e-4
            )
            assert tracer_convergence.in_list[0] == pytest.approx(
                g0_convergence.in_list[0] + g1_convergence.in_list[0], 1.0e-4
            )

        def test__no_galaxy_has_mass_profile__convergence_returned_as_zeros(
            self, sub_grid_2d_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            assert (
                tracer.convergence_2d_from(grid=sub_grid_2d_7x7).binned.native
                == np.zeros(shape=(7, 7))
            ).all()

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.1), al.Galaxy(redshift=0.2)]
            )

            assert (
                tracer.convergence_2d_from(grid=sub_grid_2d_7x7).binned.native
                == np.zeros(shape=(7, 7))
            ).all()

    class TestPotential:
        def test__galaxy_mass_sis__no_source_plane_potential(self, sub_grid_2d_7x7):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_potential = image_plane.potential_2d_from(grid=sub_grid_2d_7x7)

            tracer_potential = tracer.potential_2d_from(grid=sub_grid_2d_7x7)

            assert image_plane_potential.shape_native == (7, 7)
            assert (image_plane_potential == tracer_potential).all()

        def test__galaxy_entered_3_times__both_planes__different_potential_for_each(
            self, sub_grid_2d_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )
            g2 = al.Galaxy(
                redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=3.0)
            )

            g0_potential = g0.potential_2d_from(grid=sub_grid_2d_7x7)

            g1_potential = g1.potential_2d_from(grid=sub_grid_2d_7x7)

            g2_potential = g2.potential_2d_from(grid=sub_grid_2d_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_potential = tracer.image_plane.potential_2d_from(
                grid=sub_grid_2d_7x7
            )

            source_plane_potential = tracer.source_plane.potential_2d_from(
                grid=sub_grid_2d_7x7
            )

            tracer_potential = tracer.potential_2d_from(grid=sub_grid_2d_7x7)

            assert image_plane_potential == pytest.approx(
                g0_potential + g1_potential, 1.0e-4
            )
            assert (source_plane_potential == g2_potential).all()
            assert tracer_potential == pytest.approx(
                g0_potential + g1_potential + g2_potential, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(
            self, grid_2d_irregular_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )

            g0_potential = g0.potential_2d_from(grid=grid_2d_irregular_7x7)

            g1_potential = g1.potential_2d_from(grid=grid_2d_irregular_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_potential = tracer.image_plane.potential_2d_from(
                grid=grid_2d_irregular_7x7
            )

            source_plane_potential = tracer.source_plane.potential_2d_from(
                grid=grid_2d_irregular_7x7
            )

            tracer_potential = tracer.potential_2d_from(grid=grid_2d_irregular_7x7)

            assert image_plane_potential.in_list[0] == pytest.approx(
                g0_potential.in_list[0] + g1_potential.in_list[0], 1.0e-4
            )
            assert tracer_potential.in_list[0] == pytest.approx(
                g0_potential.in_list[0] + g1_potential.in_list[0], 1.0e-4
            )

        def test__no_galaxy_has_mass_profile__potential_returned_as_zeros(
            self, sub_grid_2d_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            assert (
                tracer.potential_2d_from(grid=sub_grid_2d_7x7).binned.native
                == np.zeros(shape=(7, 7))
            ).all()

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.1), al.Galaxy(redshift=0.2)]
            )

            assert (
                tracer.potential_2d_from(grid=sub_grid_2d_7x7).binned.native
                == np.zeros(shape=(7, 7))
            ).all()

    class TestDeflectionsOfSummedPlanes:
        def test__galaxy_mass_sis__source_plane_no_mass__deflections_is_ignored(
            self, sub_grid_2d_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(redshift=0.5)

            image_plane = al.Plane(galaxies=[g0])

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_deflections = image_plane.deflections_yx_2d_from(
                grid=sub_grid_2d_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from(
                grid=sub_grid_2d_7x7
            )

            assert tracer_deflections.shape_native == (7, 7)
            assert (image_plane_deflections == tracer_deflections).all()

        def test__galaxy_entered_3_times__different_deflections_for_each(
            self, sub_grid_2d_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )
            g2 = al.Galaxy(
                redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=3.0)
            )

            g0_deflections = g0.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

            g1_deflections = g1.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

            g2_deflections = g2.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

            image_plane_deflections = tracer.image_plane.deflections_yx_2d_from(
                grid=sub_grid_2d_7x7
            )

            source_plane_deflections = tracer.source_plane.deflections_yx_2d_from(
                grid=sub_grid_2d_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from(
                grid=sub_grid_2d_7x7
            )

            assert image_plane_deflections == pytest.approx(
                g0_deflections + g1_deflections, 1.0e-4
            )
            assert source_plane_deflections == pytest.approx(g2_deflections, 1.0e-4)
            assert tracer_deflections == pytest.approx(
                g0_deflections + g1_deflections + g2_deflections, 1.0e-4
            )

        def test__galaxy_entered_2_times__grid_is_positions(
            self, grid_2d_irregular_7x7
        ):

            g0 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
            )
            g1 = al.Galaxy(
                redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=2.0)
            )

            g0_deflections = g0.deflections_yx_2d_from(grid=grid_2d_irregular_7x7)

            g1_deflections = g1.deflections_yx_2d_from(grid=grid_2d_irregular_7x7)

            tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

            image_plane_deflections = tracer.image_plane.deflections_yx_2d_from(
                grid=grid_2d_irregular_7x7
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from(
                grid=grid_2d_irregular_7x7
            )

            assert image_plane_deflections.in_list[0][0] == pytest.approx(
                g0_deflections.in_list[0][0] + g1_deflections.in_list[0][0], 1.0e-4
            )
            assert image_plane_deflections.in_list[0][1] == pytest.approx(
                g0_deflections.in_list[0][1] + g1_deflections.in_list[0][1], 1.0e-4
            )
            assert tracer_deflections.in_list[0][0] == pytest.approx(
                g0_deflections.in_list[0][0] + g1_deflections.in_list[0][0], 1.0e-4
            )
            assert tracer_deflections.in_list[0][1] == pytest.approx(
                g0_deflections.in_list[0][1] + g1_deflections.in_list[0][1], 1.0e-4
            )

        def test__no_galaxy_has_mass_profile__deflections_returned_as_zeros(
            self, sub_grid_2d_7x7
        ):

            tracer = al.Tracer.from_galaxies(
                galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)]
            )

            tracer_deflections = tracer.deflections_of_planes_summed_from(
                grid=sub_grid_2d_7x7
            )

            assert (
                tracer_deflections.binned.native[:, :, 0] == np.zeros(shape=(7, 7))
            ).all()
            assert (
                tracer_deflections.binned.native[:, :, 1] == np.zeros(shape=(7, 7))
            ).all()

    class TestGridAtRedshift:
        def test__grid_at_redshift_from__same_as_util(self, sub_grid_2d_7x7):
            g0 = al.Galaxy(
                redshift=0.5,
                mass_profile=al.mp.SphIsothermal(
                    centre=(0.0, 0.0), einstein_radius=1.0
                ),
            )
            g1 = al.Galaxy(
                redshift=0.75,
                mass_profile=al.mp.SphIsothermal(
                    centre=(0.0, 0.0), einstein_radius=2.0
                ),
            )
            g2 = al.Galaxy(
                redshift=1.5,
                mass_profile=al.mp.SphIsothermal(
                    centre=(0.0, 0.0), einstein_radius=3.0
                ),
            )
            g3 = al.Galaxy(
                redshift=1.0,
                mass_profile=al.mp.SphIsothermal(
                    centre=(0.0, 0.0), einstein_radius=4.0
                ),
            )
            g4 = al.Galaxy(redshift=2.0)

            galaxies = [g0, g1, g2, g3, g4]

            tracer = al.Tracer.from_galaxies(galaxies=galaxies)

            grid_at_redshift_via_util = al.util.ray_tracing.grid_at_redshift_from(
                galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=0.5
            )

            grid_at_redshift = tracer.grid_at_redshift_from(
                grid=sub_grid_2d_7x7, redshift=0.5
            )

            assert grid_at_redshift == pytest.approx(grid_at_redshift_via_util, 1.0e-4)

            grid_at_redshift_via_util = al.util.ray_tracing.grid_at_redshift_from(
                galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=1.75
            )

            grid_at_redshift = tracer.grid_at_redshift_from(
                grid=sub_grid_2d_7x7, redshift=1.75
            )

            assert grid_at_redshift == pytest.approx(grid_at_redshift_via_util, 1.0e-4)

    class TestContributionMap:
        def test__contribution_maps_are_same_as_hyper_galaxy_calculation(self):

            hyper_model_image = al.Array2D.manual_native(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = al.Array2D.manual_native(
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
                tracer.contribution_map_list[0].slim
                == tracer.image_plane.contribution_map
            ).all()

            assert (
                tracer.contribution_map_list[1].slim
                == tracer.source_plane.contribution_map
            ).all()

            galaxy_0 = al.Galaxy(redshift=0.5)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

            assert (
                tracer.contribution_map == tracer.source_plane.contribution_map
            ).all()
            assert tracer.contribution_map_list[0] == None

            assert (
                tracer.contribution_map_list[1].slim
                == tracer.source_plane.contribution_map
            ).all()

            galaxy_1 = al.Galaxy(redshift=1.0)

            tracer = al.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

            assert tracer.contribution_map == None
            assert tracer.contribution_map_list[0] == None

            assert tracer.contribution_map_list[1] == None


class TestAbstractTracerData:
    def test__sparse_image_plane_grid_list_from__x2_planes(self, sub_grid_2d_7x7):
        galaxy_pix = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=np.array([[1.0, 1.0]])
            ),
            regularization=al.m.MockRegularization(),
        )
        galaxy_no_pix = al.Galaxy(redshift=0.5)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

        pixelization_grids = tracer.sparse_image_plane_grid_pg_list_from(
            grid=sub_grid_2d_7x7
        )

        assert pixelization_grids[0] == None
        assert (pixelization_grids[1] == np.array([[1.0, 1.0]])).all()

    def test__sparse_image_plane_grid_list_from__multi_plane(self, sub_grid_2d_7x7):

        galaxy_pix0 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=np.array([[1.0, 1.0]])
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_pix1 = al.Galaxy(
            redshift=2.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=np.array([[2.0, 2.0]])
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_no_pix_0 = al.Galaxy(redshift=0.25)
        galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
        galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                galaxy_pix0,
                galaxy_pix1,
                galaxy_no_pix_0,
                galaxy_no_pix_1,
                galaxy_no_pix_2,
            ]
        )

        pixelization_grids = tracer.sparse_image_plane_grid_pg_list_from(
            grid=sub_grid_2d_7x7
        )

        assert pixelization_grids[0] == None
        assert pixelization_grids[1] == None
        assert (pixelization_grids[2] == np.array([[1.0, 1.0]])).all()
        assert pixelization_grids[3] == None
        assert (pixelization_grids[4] == np.array([[2.0, 2.0]])).all()

    def test__traced_sparse_grids_list_from__x2_planes(self, sub_grid_2d_7x7):

        galaxy_pix = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=al.Grid2D.manual_native(
                    grid=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
                )
            ),
            regularization=al.m.MockRegularization(),
        )
        galaxy_no_pix = al.Galaxy(redshift=0.5)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

        traced_pixelization_grids, sparse_image_grid = tracer.traced_sparse_grid_pg_list_from(
            grid=sub_grid_2d_7x7
        )

        assert traced_pixelization_grids[0] == None
        assert (traced_pixelization_grids[1] == np.array([[1.0, 0.0]])).all()

        galaxy_no_pix = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
        )

        galaxy_pix_0 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=al.Grid2D.manual_native(
                    grid=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
                )
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_pix_1 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=al.Grid2D.manual_native(
                    grid=[[[2.0, 0.0]]], pixel_scales=(1.0, 1.0)
                )
            ),
            regularization=al.m.MockRegularization(),
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1]
        )

        traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer.traced_sparse_grid_pg_list_from(
            grid=sub_grid_2d_7x7
        )

        assert traced_sparse_grids_list_of_planes[0] == None
        assert traced_sparse_grids_list_of_planes[1][0] == pytest.approx(
            np.array([[1.0 - 0.5, 0.0]]), 1.0e-4
        )
        assert traced_sparse_grids_list_of_planes[1][1] == pytest.approx(
            np.array([[2.0 - 0.5, 0.0]]), 1.0e-4
        )

    def test__traced_sparse_grids_list_from__multi_plane(self, sub_grid_2d_7x7):

        galaxy_pix_0 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=al.Grid2D.manual_native(
                    grid=[[[1.0, 1.0]]], pixel_scales=(1.0, 1.0)
                )
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_pix_1 = al.Galaxy(
            redshift=2.0,
            pixelization=al.m.MockPixelization(
                data_pixelization_grid=al.Grid2D.manual_native(
                    grid=[[[2.0, 2.0]]], pixel_scales=(1.0, 1.0)
                )
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_no_pix_0 = al.Galaxy(
            redshift=0.25,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
        )
        galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
        galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                galaxy_pix_0,
                galaxy_pix_1,
                galaxy_no_pix_0,
                galaxy_no_pix_1,
                galaxy_no_pix_2,
            ]
        )

        traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer.traced_sparse_grid_pg_list_from(
            grid=sub_grid_2d_7x7
        )

        traced_grid_pix_0 = tracer.traced_grid_list_from(grid=np.array([[1.0, 1.0]]))[2]
        traced_grid_pix_1 = tracer.traced_grid_list_from(grid=np.array([[2.0, 2.0]]))[4]

        assert traced_sparse_grids_list_of_planes[0] == None
        assert traced_sparse_grids_list_of_planes[1] == None
        assert (traced_sparse_grids_list_of_planes[2][0] == traced_grid_pix_0).all()
        assert traced_sparse_grids_list_of_planes[3] == None
        assert (traced_sparse_grids_list_of_planes[4][0] == traced_grid_pix_1).all()

    def test__linear_obj_list_from__x2_planes(self, sub_grid_2d_7x7):

        galaxy_no_pix = al.Galaxy(redshift=0.5)

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

        mappers_of_planes = tracer.linear_obj_list_from(grid=sub_grid_2d_7x7)
        assert mappers_of_planes == []

        galaxy_no_pix = al.Galaxy(redshift=0.5)
        galaxy_pix_0 = al.Galaxy(
            redshift=0.5,
            pixelization=al.m.MockPixelization(
                mapper=1, data_pixelization_grid=sub_grid_2d_7x7
            ),
            regularization=al.m.MockRegularization(),
        )

        galaxy_pix_1 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                mapper=2, data_pixelization_grid=sub_grid_2d_7x7
            ),
            regularization=al.m.MockRegularization(),
        )
        galaxy_pix_2 = al.Galaxy(
            redshift=1.0,
            pixelization=al.m.MockPixelization(
                mapper=3, data_pixelization_grid=sub_grid_2d_7x7
            ),
            regularization=al.m.MockRegularization(),
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1, galaxy_pix_2]
        )

        linear_obj_list = tracer.linear_obj_list_from(grid=sub_grid_2d_7x7)

        assert linear_obj_list == [1, 2, 3]

    def test__linear_obj_list_from__multi_plane(self, sub_grid_2d_7x7):

        galaxy_no_pix_0 = al.Galaxy(
            redshift=0.25,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
        )
        galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
        galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

        galaxy_pix_0 = al.Galaxy(
            redshift=0.75,
            pixelization=al.m.MockPixelization(
                mapper=1, data_pixelization_grid=sub_grid_2d_7x7
            ),
            regularization=al.m.MockRegularization(),
        )
        galaxy_pix_1 = al.Galaxy(
            redshift=2.0,
            pixelization=al.m.MockPixelization(
                mapper=2, data_pixelization_grid=sub_grid_2d_7x7
            ),
            regularization=al.m.MockRegularization(),
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                galaxy_no_pix_0,
                galaxy_no_pix_1,
                galaxy_no_pix_2,
                galaxy_pix_0,
                galaxy_pix_1,
            ]
        )

        linear_obj_list = tracer.linear_obj_list_from(grid=sub_grid_2d_7x7)

        assert linear_obj_list == [1, 2]

    def test__inversion_imaging_from(self, sub_grid_2d_7x7, masked_imaging_7x7):

        pix = al.pix.Rectangular(shape=(3, 3))
        reg = al.reg.Constant(coefficient=0.0)

        g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

        tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

        inversion = tracer.inversion_imaging_from(
            grid=sub_grid_2d_7x7,
            image=masked_imaging_7x7.image,
            noise_map=masked_imaging_7x7.noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            settings_pixelization=al.SettingsPixelization(use_border=False),
            settings_inversion=al.SettingsInversion(use_w_tilde=False),
        )

        assert inversion.mapped_reconstructed_image == pytest.approx(
            masked_imaging_7x7.image, 1.0e-2
        )

    def test__inversion_interferometer_from(self, sub_grid_2d_7x7, interferometer_7):

        interferometer_7.data = al.Visibilities.ones(shape_slim=(7,))

        pix = al.pix.Rectangular(shape=(7, 7))
        reg = al.reg.Constant(coefficient=0.0)

        g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

        tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

        inversion = tracer.inversion_interferometer_from(
            grid=sub_grid_2d_7x7,
            visibilities=interferometer_7.visibilities,
            noise_map=interferometer_7.noise_map,
            transformer=interferometer_7.transformer,
            w_tilde=interferometer_7.w_tilde,
            settings_pixelization=al.SettingsPixelization(use_border=False),
        )

        assert inversion.reconstruction[0] == pytest.approx(-0.2662, 1.0e-4)

    def test__hyper_noise_map_list_from(self, sub_grid_2d_7x7):

        noise_map_1d = al.Array2D.manual_native(
            array=[[5.0, 3.0, 1.0]], pixel_scales=1.0
        )

        hyper_model_image = al.Array2D.manual_native(
            array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
        )
        hyper_galaxy_image = al.Array2D.manual_native(
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

        hyper_noise_map_0 = plane_0.hyper_noise_map_from(noise_map=noise_map_1d)
        hyper_noise_map_1 = plane_1.hyper_noise_map_from(noise_map=noise_map_1d)

        tracer = al.Tracer(planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15)

        hyper_noise_maps = tracer.hyper_noise_map_list_from(noise_map=noise_map_1d)

        assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()
        assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()
        assert (hyper_noise_maps[2].slim == np.zeros(shape=(3, 1))).all()

        hyper_noise_map = tracer.hyper_noise_map_from(noise_map=noise_map_1d)

        assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

        tracer = al.Tracer.from_galaxies(
            galaxies=[galaxy_0, galaxy_1], cosmology=cosmo.Planck15
        )

        hyper_noise_maps = tracer.hyper_noise_map_list_from(noise_map=noise_map_1d)

        assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()
        assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()


class TestTracer:
    def test__x2_planes__no_galaxy__all_deflections_are_zeros(
        self, sub_grid_2d_7x7_simple
    ):

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        traced_deflections_between_planes = tracer.deflections_between_planes_from(
            grid=sub_grid_2d_7x7_simple, plane_i=0, plane_j=0
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

        traced_deflections_between_planes = tracer.deflections_between_planes_from(
            grid=sub_grid_2d_7x7_simple, plane_i=0, plane_j=1
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
        self, sub_grid_2d_7x7_simple, gal_x1_mp
    ):

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_mp, al.Galaxy(redshift=1.0)])

        traced_deflections_between_planes = tracer.deflections_between_planes_from(
            grid=sub_grid_2d_7x7_simple, plane_i=0, plane_j=1
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
        self, sub_grid_2d_7x7_simple, gal_x1_mp
    ):

        tracer = al.Tracer.from_galaxies(
            galaxies=[gal_x1_mp, gal_x1_mp, al.Galaxy(redshift=1.0)]
        )

        traced_deflections_between_planes = tracer.deflections_between_planes_from(
            grid=sub_grid_2d_7x7_simple, plane_i=0, plane_j=1
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


class TestTracerFixedSlices:
    def test__6_galaxies__tracer_planes_are_correct(self, sub_grid_2d_7x7):
        lens_g0 = al.Galaxy(redshift=0.5)
        source_g0 = al.Galaxy(redshift=2.0)
        los_g0 = al.Galaxy(redshift=0.1)
        los_g1 = al.Galaxy(redshift=0.2)
        los_g2 = al.Galaxy(redshift=0.4)
        los_g3 = al.Galaxy(redshift=0.6)

        tracer = al.Tracer.sliced_tracer_from(
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
        self, sub_grid_2d_7x7_simple
    ):

        lens_g0 = al.Galaxy(
            redshift=0.5, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        source_g0 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        los_g0 = al.Galaxy(
            redshift=0.1, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        los_g1 = al.Galaxy(
            redshift=0.2, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        los_g2 = al.Galaxy(
            redshift=0.4, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        los_g3 = al.Galaxy(
            redshift=0.6, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )

        tracer = al.Tracer.sliced_tracer_from(
            lens_galaxies=[lens_g0],
            line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
            source_galaxies=[source_g0],
            planes_between_lenses=[1, 1],
            cosmology=cosmo.Planck15,
        )

        traced_grids = tracer.traced_grid_list_from(grid=sub_grid_2d_7x7_simple)

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

        #  galaxies in this plane, so multiply by 3

        defl11 = 3.0 * lens_g0.deflections_yx_2d_from(
            grid=np.array([[(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]])
        )
        defl12 = 3.0 * lens_g0.deflections_yx_2d_from(
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


class TestExtractAttribute:
    def test__extract_attribute(self):

        g0 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
        )
        g1 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
        )
        g2 = al.Galaxy(
            redshift=0.5,
            mp_0=al.m.MockMassProfile(value=0.7),
            mp_1=al.m.MockMassProfile(value=0.6),
        )

        plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
        plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

        assert values == None

        plane_0 = al.Plane(galaxies=[g0], redshift=None)
        plane_1 = al.Plane(galaxies=[g1], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

        assert values.in_list == [0.9, 0.8]

        values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value1")

        assert values.in_list == [(1.0, 1.0), (2.0, 2.0)]

        plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
        plane_1 = al.Plane(galaxies=[g2], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

        assert values.in_list == [0.9, 0.8, 0.7, 0.6]

        tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")

    def test__extract_attributes_of_planes(self):

        g0 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
        )
        g1 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
        )
        g2 = al.Galaxy(
            redshift=0.5,
            mp_0=al.m.MockMassProfile(value=0.7),
            mp_1=al.m.MockMassProfile(value=0.6),
        )

        plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
        plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attributes_of_planes(
            cls=al.mp.MassProfile, attr_name="value"
        )

        assert values == [None, None]

        plane_0 = al.Plane(galaxies=[g0], redshift=None)
        plane_1 = al.Plane(galaxies=[g1], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attributes_of_planes(
            cls=al.mp.MassProfile, attr_name="value"
        )

        assert values[0].in_list == [0.9]
        assert values[1].in_list == [0.8]

        values = tracer.extract_attributes_of_planes(
            cls=al.mp.MassProfile, attr_name="value1"
        )

        assert values[0].in_list == [(1.0, 1.0)]
        assert values[1].in_list == [(2.0, 2.0)]

        plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
        plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
        plane_2 = al.Plane(galaxies=[g2], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1, plane_2], cosmology=None)

        values = tracer.extract_attributes_of_planes(
            cls=al.mp.MassProfile, attr_name="value", filter_nones=False
        )

        assert values[0].in_list == [0.9, 0.8]
        assert values[1] == None
        assert values[2].in_list == [0.7, 0.6]

        values = tracer.extract_attributes_of_planes(
            cls=al.mp.MassProfile, attr_name="value", filter_nones=True
        )

        assert values[0].in_list == [0.9, 0.8]
        assert values[1].in_list == [0.7, 0.6]

        tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")

    def test__extract_attributes_of_galaxies(self):

        g0 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
        )
        g1 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
        )
        g2 = al.Galaxy(
            redshift=0.5,
            mp_0=al.m.MockMassProfile(value=0.7),
            mp_1=al.m.MockMassProfile(value=0.6),
        )

        plane_0 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
        plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=1.0)], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attributes_of_galaxies(
            cls=al.mp.MassProfile, attr_name="value"
        )

        assert values == [None, None]

        plane_0 = al.Plane(galaxies=[g0], redshift=None)
        plane_1 = al.Plane(galaxies=[g1], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=None)

        values = tracer.extract_attributes_of_galaxies(
            cls=al.mp.MassProfile, attr_name="value"
        )

        assert values[0].in_list == [0.9]
        assert values[1].in_list == [0.8]

        values = tracer.extract_attributes_of_galaxies(
            cls=al.mp.MassProfile, attr_name="value1"
        )

        assert values[0].in_list == [(1.0, 1.0)]
        assert values[1].in_list == [(2.0, 2.0)]

        plane_0 = al.Plane(galaxies=[g0, g1], redshift=None)
        plane_1 = al.Plane(galaxies=[al.Galaxy(redshift=0.5)], redshift=None)
        plane_2 = al.Plane(galaxies=[g2], redshift=None)

        tracer = al.Tracer(planes=[plane_0, plane_1, plane_2], cosmology=None)

        values = tracer.extract_attributes_of_galaxies(
            cls=al.mp.MassProfile, attr_name="value", filter_nones=False
        )

        assert values[0].in_list == [0.9]
        assert values[1].in_list == [0.8]
        assert values[2] == None
        assert values[3].in_list == [0.7, 0.6]

        values = tracer.extract_attributes_of_galaxies(
            cls=al.mp.MassProfile, attr_name="value", filter_nones=True
        )

        assert values[0].in_list == [0.9]
        assert values[1].in_list == [0.8]
        assert values[2].in_list == [0.7, 0.6]

        tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")

    def test__extract_profile(self):

        g0 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
        )
        g1 = al.Galaxy(
            redshift=0.5, mp_1=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
        )
        g2 = al.Galaxy(
            redshift=1.0,
            mp_2=al.m.MockMassProfile(value=0.7),
            mp_3=al.m.MockMassProfile(value=0.6),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2], cosmology=None)

        profile = tracer.extract_profile(profile_name="mp_0")

        assert profile.value == 0.9

        profile = tracer.extract_profile(profile_name="mp_3")

        assert profile.value == 0.6

    def test__extract_plane_index_of_profile(self):

        g0 = al.Galaxy(
            redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
        )
        g1 = al.Galaxy(
            redshift=0.75, mp_1=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
        )
        g2 = al.Galaxy(
            redshift=1.0,
            mp_2=al.m.MockMassProfile(value=0.7),
            mp_3=al.m.MockMassProfile(value=0.6),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2], cosmology=None)

        plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_0")

        assert plane_index == 0

        plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_1")

        assert plane_index == 1

        plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_3")

        assert plane_index == 2


class TestSNRLightProfiles:
    def test__signal_to_noise_via_simulator_correct(self):

        background_sky_level = 10.0
        exposure_time = 300.0

        grid = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

        mass = al.mp.SphIsothermal(einstein_radius=1.0)

        sersic = al.lp_snr.EllSersic(signal_to_noise_ratio=10.0, effective_radius=0.01)

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(redshift=0.5, mass=mass),
                al.Galaxy(redshift=1.0, light=sersic),
            ]
        )

        psf = al.Kernel2D.manual_native(array=[[1.0]], pixel_scales=1.0)

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=exposure_time,
            noise_seed=1,
            background_sky_level=background_sky_level,
        )

        imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

        assert 8.0 < imaging.signal_to_noise_map.native[0, 1] < 12.0
        assert 8.0 < imaging.signal_to_noise_map.native[1, 0] < 12.0
        assert 8.0 < imaging.signal_to_noise_map.native[1, 2] < 12.0
        assert 8.0 < imaging.signal_to_noise_map.native[2, 1] < 12.0


class TestRegression:
    def test__centre_of_profile_in_right_place(self):

        grid = al.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = tracer.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] > 0
        assert deflections.native[2, 4, 0] < 0
        assert deflections.native[1, 4, 1] > 0
        assert deflections.native[1, 3, 1] < 0

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = tracer.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] > 0
        assert deflections.native[2, 4, 0] < 0
        assert deflections.native[1, 4, 1] > 0
        assert deflections.native[1, 3, 1] < 0

        grid = al.Grid2DIterate.uniform(
            shape_native=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=al.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = tracer.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] >= -1e-8
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        convergence = tracer.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = tracer.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = tracer.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] >= -1e-8
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0


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

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_lp])

        image = tracer.image_2d_from(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid2D.from_mask(mask=mask_sub_2)
        image_sub_2 = tracer.image_2d_from(grid=grid_sub_2).binned

        assert (image == image_sub_2).all()

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllSersic(centre=(0.08, 0.08), intensity=1.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy])

        image = tracer.image_2d_from(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = al.Grid2D.from_mask(mask=mask_sub_4)
        image_sub_4 = tracer.image_2d_from(grid=grid_sub_4).binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid2D.from_mask(mask=mask_sub_8)
        image_sub_8 = tracer.image_2d_from(grid=grid_sub_8).binned

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

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_lp])

        images = tracer.image_2d_list_from(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid2D.from_mask(mask=mask_sub_2)
        image_sub_2 = tracer.image_2d_from(grid=grid_sub_2).binned

        assert (images[0] == image_sub_2).all()

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllSersic(centre=(0.08, 0.08), intensity=1.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy])

        images = tracer.image_2d_list_from(grid=grid)

        mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid2D.from_mask(mask=mask_sub_8)
        image_sub_8 = tracer.image_2d_from(grid=grid_sub_8).binned

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

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        deflections = tracer.deflections_yx_2d_from(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = al.Grid2D.from_mask(mask=mask_sub_2)
        deflections_sub_2 = tracer.deflections_yx_2d_from(grid=grid_sub_2).binned

        assert (deflections == deflections_sub_2).all()

        grid = al.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy, al.Galaxy(redshift=1.0)])

        deflections = tracer.deflections_yx_2d_from(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = al.Grid2D.from_mask(mask=mask_sub_4)
        deflections_sub_4 = tracer.deflections_yx_2d_from(grid=grid_sub_4).binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
        grid_sub_8 = al.Grid2D.from_mask(mask=mask_sub_8)
        deflections_sub_8 = galaxy.deflections_yx_2d_from(grid=grid_sub_8).binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]
