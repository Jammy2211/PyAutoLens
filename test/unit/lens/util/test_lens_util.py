import numpy as np
import pytest

from autolens import exc
from autolens.data.array.util import mapping_util
from autolens.model.profiles import light_profiles as lp
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util
from autolens.lens import plane as pl
from autolens.lens.util import lens_util

class TestPlaneImageFromGrid:

    def test__3x3_grid__extracts_max_min_coordinates__creates_regular_grid_including_half_pixel_offset_from_edge(self):

        galaxy = g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = lens_util.plane_image_of_galaxies_from_grid(
            shape=(3, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array(
            [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
             [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
             [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self):

        galaxy = g.Galaxy(redshift=0.5,
                          light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5], [0.1, -0.1], [-1.0, 0.6], [1.4, -1.3], [1.5, 1.5]])

        plane_image = lens_util.plane_image_of_galaxies_from_grid(
            shape=(3, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array(
            [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
             [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
             [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__2x3_grid__shape_change_correct_and_coordinates_shift(self):

        galaxy = g.Galaxy(redshift=0.5,
                          light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = lens_util.plane_image_of_galaxies_from_grid(
            shape=(2, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array(
            [[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
             [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(2,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x2_grid__shape_change_correct_and_coordinates_shift(self):

        galaxy = g.Galaxy(redshift=0.5,
                          light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = lens_util.plane_image_of_galaxies_from_grid(
            shape=(3, 2), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array(
            [[-1.0, -0.75], [-1.0, 0.75],
             [0.0, -0.75], [0.0, 0.75],
             [1.0, -0.75], [1.0, 0.75]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,2))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x3_grid__buffer_aligns_two_grids(self):

        galaxy = g.Galaxy(redshift=0.5,
                          light=lp.EllipticalSersic(intensity=1.0))

        grid_without_buffer = np.array([[-1.48, -1.48], [1.48, 1.48]])

        plane_image = lens_util.plane_image_of_galaxies_from_grid(
            shape=(3, 3), grid=grid_without_buffer, galaxies=[galaxy], buffer=0.02)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array(
            [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
             [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
             [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()


class TestSetupTracedGrid:

    def test__simple_sis_model__deflection_angles(
            self, grid_stack_simple, gal_x1_mp):

        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x1_mp])

        grid_traced = lens_util.traced_collection_for_deflections(grid_stack_simple, deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)

    def test_two_identical_lenses__deflection_angles_double(
            self, grid_stack_simple, gal_x1_mp):
        
        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x1_mp, gal_x1_mp])

        grid_traced = lens_util.traced_collection_for_deflections(grid_stack_simple, deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3)

    def test_one_lens_with_double_identical_mass_profiles__deflection_angles_double(
            self, grid_stack_simple, gal_x2_mp):

        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x2_mp])

        grid_traced = lens_util.traced_collection_for_deflections(
            grid_stack=grid_stack_simple, deflections=deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-3)


class TestPlaneRedshifts:

    def test__from_galaxies__3_galaxies_reordered_in_ascending_redshift(self):

        galaxies = [g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 1.0, 2.0]

    def test_from_galaxies__3_galaxies_two_same_redshift_planes_redshift_order_is_size_2_with_redshifts(self):

        galaxies = [g.Galaxy(redshift=1.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 1.0]

    def test__from_galaxies__6_galaxies_producing_4_planes(self):

        g0 = g.Galaxy(redshift=1.0)
        g1 = g.Galaxy(redshift=1.0)
        g2 = g.Galaxy(redshift=0.1)
        g3 = g.Galaxy(redshift=1.05)
        g4 = g.Galaxy(redshift=0.95)
        g5 = g.Galaxy(redshift=1.05)

        galaxies = [g0, g1, g2, g3, g4, g5]

        ordered_plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 0.95, 1.0, 1.05]

    def test__from_main_plane_redshifts_and_slices(self):

        ordered_plane_redshifts = lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
            lens_redshifts=[1.0], source_plane_redshift=3.0, planes_between_lenses=[1, 1])

        assert ordered_plane_redshifts == [0.5, 1.0, 2.0]

    def test__different_number_of_slices_between_planes(self):

        ordered_plane_redshifts = lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
            lens_redshifts=[1.0], source_plane_redshift=2.0, planes_between_lenses=[2, 3])

        assert ordered_plane_redshifts == [(1.0/3.0), (2.0/3.0), 1.0, 1.25, 1.5, 1.75]

    def test__if_number_of_input_slices_is_not_equal_to_number_of_plane_intervals__raises_errror(self):

        with pytest.raises(exc.RayTracingException):
            lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
                lens_redshifts=[1.0], source_plane_redshift=2.0, planes_between_lenses=[2, 3, 1])

        with pytest.raises(exc.RayTracingException):
            lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
                lens_redshifts=[1.0], source_plane_redshift=2.0, planes_between_lenses=[2])

        with pytest.raises(exc.RayTracingException):
            lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
                lens_redshifts=[1.0, 3.0], source_plane_redshift=2.0, planes_between_lenses=[2])


class TestGalaxyOrdering:

    def test__3_galaxies_reordered_in_ascending_redshift__planes_match_galaxy_redshifts(self):

        galaxies = [g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = [0.1, 1.0, 2.0]

        galaxies_in_redshift_ordered_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts)

        assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
        assert galaxies_in_redshift_ordered_planes[1][0].redshift == 1.0
        assert galaxies_in_redshift_ordered_planes[2][0].redshift == 2.0

    def test_3_galaxies_x2_same_redshift__order_is_size_2_with_redshifts__plane_match_galaxy_redshifts(self):

        galaxies = [g.Galaxy(redshift=1.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = [0.1, 1.0]

        galaxies_in_redshift_ordered_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts)

        assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
        assert galaxies_in_redshift_ordered_planes[1][0].redshift == 1.0
        assert galaxies_in_redshift_ordered_planes[1][1].redshift == 1.0

    def test__6_galaxies_producing_4_planes__galaxy_redshift_match_planes(self):

        g0 = g.Galaxy(redshift=1.0)
        g1 = g.Galaxy(redshift=1.0)
        g2 = g.Galaxy(redshift=0.1)
        g3 = g.Galaxy(redshift=1.05)
        g4 = g.Galaxy(redshift=0.95)
        g5 = g.Galaxy(redshift=1.05)

        galaxies = [g0, g1, g2, g3, g4, g5]

        ordered_plane_redshifts = [0.1, 0.95, 1.0, 1.05]

        galaxies_in_redshift_ordered_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts)

        assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
        assert galaxies_in_redshift_ordered_planes[1][0].redshift == 0.95
        assert galaxies_in_redshift_ordered_planes[2][0].redshift == 1.0
        assert galaxies_in_redshift_ordered_planes[2][1].redshift == 1.0
        assert galaxies_in_redshift_ordered_planes[3][0].redshift == 1.05
        assert galaxies_in_redshift_ordered_planes[3][1].redshift == 1.05

        assert galaxies_in_redshift_ordered_planes[0] == [g2]
        assert galaxies_in_redshift_ordered_planes[1] == [g4]
        assert galaxies_in_redshift_ordered_planes[2] == [g0, g1]
        assert galaxies_in_redshift_ordered_planes[3] == [g3, g5]

    def test__galaxy_redshifts_dont_match_plane_redshifts__tied_to_nearest_plane(self):

        ordered_plane_redshifts = [0.5, 1.0, 2.0, 3.0]

        galaxies = [g.Galaxy(redshift=0.2), g.Galaxy(redshift=0.4), g.Galaxy(redshift=0.8), g.Galaxy(redshift=1.2),
                    g.Galaxy(redshift=2.9)]

        galaxies_in_redshift_ordered_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts)

        assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.2
        assert galaxies_in_redshift_ordered_planes[0][1].redshift == 0.4
        assert galaxies_in_redshift_ordered_planes[1][0].redshift == 0.8
        assert galaxies_in_redshift_ordered_planes[1][1].redshift == 1.2
        assert galaxies_in_redshift_ordered_planes[2] == []
        assert galaxies_in_redshift_ordered_planes[3][0].redshift == 2.9

    def test__different_number_of_slices_between_planes(self):

        ordered_plane_redshifts = [(1.0/3.0), (2.0/3.0), 1.0, 1.25, 1.5, 1.75, 2.0]

        galaxies = [g.Galaxy(redshift=0.1), g.Galaxy(redshift=0.2), g.Galaxy(redshift=1.25), g.Galaxy(redshift=1.35),
                    g.Galaxy(redshift=1.45), g.Galaxy(redshift=1.55), g.Galaxy(redshift=1.9)]

        galaxies_in_redshift_ordered_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts)

        assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
        assert galaxies_in_redshift_ordered_planes[0][1].redshift == 0.2
        assert galaxies_in_redshift_ordered_planes[1] == []
        assert galaxies_in_redshift_ordered_planes[2] == []
        assert galaxies_in_redshift_ordered_planes[3][0].redshift == 1.25
        assert galaxies_in_redshift_ordered_planes[3][1].redshift == 1.35
        assert galaxies_in_redshift_ordered_planes[4][0].redshift == 1.45
        assert galaxies_in_redshift_ordered_planes[4][1].redshift == 1.55
        assert galaxies_in_redshift_ordered_planes[6][0].redshift == 1.9


class TestComputeDeflections:

    def test__if_plane_is_last_plane_is_false_else_is_true(self):

        assert lens_util.compute_deflections_at_next_plane(plane_index=0, total_planes=4) == True
        assert lens_util.compute_deflections_at_next_plane(plane_index=2, total_planes=4) == True
        assert lens_util.compute_deflections_at_next_plane(plane_index=3, total_planes=4) == False
        assert lens_util.compute_deflections_at_next_plane(plane_index=3, total_planes=5) == True


class TestScaledDeflections:

    def test__deflection_stack_is_scaled_by_scaling_factor(self, grid_stack_simple, gal_x1_mp):

        plane = pl.Plane(galaxies=[gal_x1_mp], grid_stack=grid_stack_simple)

        scaled_deflection_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
            plane=plane, scaling_factor=1.0)

        assert (scaled_deflection_stack.regular == plane.deflections_stack.regular).all()
        assert (scaled_deflection_stack.sub == plane.deflections_stack.sub).all()
        assert (scaled_deflection_stack.blurring == plane.deflections_stack.blurring).all()

        scaled_deflection_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
            plane=plane, scaling_factor=2.0)

        assert (scaled_deflection_stack.regular == 2.0 * plane.deflections_stack.regular).all()
        assert (scaled_deflection_stack.sub == 2.0 * plane.deflections_stack.sub).all()
        assert (scaled_deflection_stack.blurring == 2.0 * plane.deflections_stack.blurring).all()


class TestGridStackDeflections:

    def test__grid_stack_has_deflections_subtracted_from_it(self, grid_stack_simple, gal_x1_mp):

        plane = pl.Plane(galaxies=[gal_x1_mp], grid_stack=grid_stack_simple)

        deflection_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
            plane=plane, scaling_factor=3.0)

        traced_grid_stack = lens_util.grid_stack_from_deflections_stack(
            grid_stack=grid_stack_simple, deflections_stack=deflection_stack)

        assert (traced_grid_stack.regular == grid_stack_simple.regular - deflection_stack.regular).all()
        assert (traced_grid_stack.sub == grid_stack_simple.sub - deflection_stack.sub).all()
        assert (traced_grid_stack.blurring == grid_stack_simple.blurring - deflection_stack.blurring).all()