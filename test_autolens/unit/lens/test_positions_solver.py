import autolens as al
from autolens.lens import positions_solver as pos
import pytest

def test__positions_for_simple_mass_profiles():

    grid = al.Grid.uniform(shape_2d=(10000, 10000), pixel_scales=0.01)

    sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    solver = al.PositionsSolver(grid=grid)

    positions = solver.image_plane_positions_from(lensing_obj=sis, source_plane_coordinate=(0.0, 0.11))

    print(positions)
    
    
class TestPeakPixels:
    def test__simple_arrays(self):

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[2, 2]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1], [2, 3]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[1, 3], [2, 1]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[2, 3]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1], [3, 3]]

    def test__simple_arrays_with_mask(self):

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d, mask=mask)

        assert peak_pixels == [[2, 3]]

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d, mask=mask)

        assert peak_pixels == []

        array = al.Array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = pos.peak_pixels_from(array_2d=array.in_2d, mask=mask)

        assert peak_pixels == [[3, 3]]


class TestTroughPixels:
    def test__simple_arrays(self):

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 1.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[2, 2]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1], [2, 3]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 2.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[1, 3], [2, 1]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 7.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [4.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 8.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[2, 3]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 0.1, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 0.1, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1], [3, 3]]

    def test__simple_arrays_with_mask(self):

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d, mask=mask)

        assert trough_pixels == [[2, 3]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d, mask=mask)

        assert trough_pixels == []

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 0.1, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = pos.trough_pixels_from(array_2d=array.in_2d, mask=mask)

        assert trough_pixels == [[3, 3]]
        

class TestPositionsAtCoordinate:
    def test__uniform_grid__locates_pixels_correctly(self):

        grid = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0)

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.3, 0.3)
        )

        assert pixels_at_coordinate == [(1, 2), (1, 3), (2, 2), (2, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(-0.3, 0.3)
        )

        assert pixels_at_coordinate == [(2, 2), (2, 3), (3, 2), (3, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.6, 0.6)
        )

        assert pixels_at_coordinate == [(1, 2), (1, 3), (2, 2), (2, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(1.1, 1.1)
        )

        assert pixels_at_coordinate == [(1, 3)]

    def test__uniform_grid__mask_remove_points(self):

        grid = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0)

        mask = al.Mask.manual(
            mask=[
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.3, 0.3), mask=mask
        )

        assert pixels_at_coordinate == [(1, 3)]

    def test__non_uniform_grid__locates_multiple_pixels_correctly(self):

        grid = al.Grid.manual_2d(
            grid=[
                [
                    [3.0, 1.0],
                    [0.0, 0.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 3.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [3.0, 1.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
            pixel_scales=1.0,
        )

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(2.0, 2.0)
        )

        assert pixels_at_coordinate == [(1, 1), (4, 5)]