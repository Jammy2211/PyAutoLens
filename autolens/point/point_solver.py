from functools import partial
import numpy as np

import autoarray as aa
import autogalaxy as ag


class AbstractPointSolver:
    def __init__(
        self,
        use_upscaling=True,
        upscale_factor=2,
        magnification_threshold=0.1,
        distance_to_source_centre=None,
        distance_to_mass_profile_centre=None,
    ):

        self.use_upscaling = use_upscaling
        self.upscale_factor = upscale_factor
        self.magnification_threshold = magnification_threshold
        self.distance_to_source_centre = distance_to_source_centre
        self.distance_to_mass_profile_centre = distance_to_mass_profile_centre

    def grid_with_points_below_magnification_threshold_removed(
        self, lensing_obj, deflections_func, grid
    ):

        magnifications = np.abs(
            lensing_obj.magnification_2d_via_hessian_from(
                grid=grid, buffer=grid.pixel_scale, deflections_func=deflections_func
            )
        )

        grid_mag = []

        for index, magnification in enumerate(magnifications):
            if magnification > self.magnification_threshold:
                grid_mag.append(grid[index, :])

        return aa.Grid2DIrregularUniform(grid=grid_mag, pixel_scales=grid.pixel_scales)

    def grid_with_coordinates_to_mass_profile_centre_removed_from(
        self, lensing_obj, grid
    ):
        """Remove all coordinates from a grid which are within the distance_to_mass_profile_centre attribute of any
        mass profile of the lensing object.

        The `PositionFinder` often finds multiple unphyiscal solutions near a mass profile due to the high levels of
        demagnification. These are typically not observable in real galaxies and thus may benefit from being removed
        from the PositionFiner.

        The positions are removed by computing the distance between all grid points and the mass profile centres of
        every mass profile in the lensing object.

        Parameters
        ----------
        lensing_obj
            An object which has a `deflection_2d_from` method for performing lensing calculations, for example a
            `MassProfile`, _Galaxy_, `Plane` or `Tracer`.
        grid : autoarray.Grid2DIrregularUniform or ndarray
            A gridd of (y,x) Cartesian coordinates for which their distances to the mass profile centres are computed,
            with points within the threshold removed.
        """
        if self.distance_to_mass_profile_centre is not None:

            pixel_scales = grid.pixel_scales

            centres = lensing_obj.extract_attribute(
                cls=ag.mp.MassProfile, attr_name="centre"
            )

            for centre in centres.in_list:

                distances_1d = np.sqrt(
                    np.square(grid[:, 0] - centre[0])
                    + np.square(grid[:, 1] - centre[1])
                )

                grid = grid_outside_distance_mask_from(
                    distances_1d=distances_1d,
                    grid_slim=grid,
                    outside_distance=self.distance_to_mass_profile_centre,
                )

            return aa.Grid2DIrregularUniform(grid=grid, pixel_scales=pixel_scales)

        return grid

    def grid_buffed_and_upscaled_around_coordinate_from(
        self, coordinate, pixel_scales, buffer, upscale_factor
    ):
        """
        For an input (y,x) Catersian coordinate create a buffed and upscaled square grid of (y,x) coordinates where:

            - The new grid of coordinates are buffed. For example, if buffer=1, the new grid will correspond to a 3x3 grid
              of coordinates centred on the input (y,x) value with spacings defined by the input pixel_scales.

            - The new grid is upscaled. For example, if upscale=2, the new grid will be at x2 the resolution of the input
              pixel_scale.

            Buffing and upscaling work together, so a buffer=2 and upscale=2 will produce a new 6x6 grid centred around the
            input coordinate.

            The `PositionFinder` works by locating pixels that trace closer to the source galaxy than neighboring pixels
            and iteratively refining the grid to find pixels that trace close at higher and higher resolutions. This
            function is core to producing these upscaled grids.

            Parameters
            ----------
            coordinate : (float, float)
                The (y,x) Cartesian coordinates aroun which the buffed and upscaled grid is created.
            pixel_scales : (float, float)
                The pixel-scale resolution of the buffed and upscaled grid that is formed around the input coordinate. If
                upscale > 1, the pixel_scales are reduced to pixel_scale / upscale_factor.
            buffer
                The number of pixels around the central (y,x) coordinate that the grid is computed on, i.e. how much it is
                buffed. A buffer of 1 puts 1 pixel in every direction around the (y,x) coordinate, creating a 3x3 grid. A
                buffer=2 places two pixels around it in every direction, creating a 5x5 grid. And so on.
            upscale_factor
                The factor by which the resolution of the grid is increased relative to the input pixel-scales.
        """

        if self.use_upscaling:
            upscale_factor = upscale_factor
        else:
            upscale_factor = 1

        grid_buffed = grid_buffed_around_coordinate_from(
            coordinate=coordinate,
            pixel_scales=pixel_scales,
            buffer=buffer,
            upscale_factor=upscale_factor,
        )

        return aa.Grid2DIrregularUniform(
            grid=grid_buffed,
            pixel_scales=(
                pixel_scales[0] / upscale_factor,
                pixel_scales[1] / upscale_factor,
            ),
        )

    def grid_peaks_from(self, deflections_func, grid, source_plane_coordinate):
        """
        Find the 'peaks' of a grid of coordinates, where a peak corresponds to a (y,x) coordinate on the grid which
        traces closer to the input (y,x) source-plane coordinate than any of its 8 adjacent neighbors. This is
        performed by:

         1) Computing the deflection angle of every (y,x) coordinate on the grid using the input lensing object.
         2) Ray tracing these coordinates to the source-plane.
         3) Computing their distance to the centre of the source in the source-plane.
         4) Finding pixels whose source-plane distance is lower than all 8 neighboring pixels.

        The `PositionFinder` works by locating pixels that trace closer to the source galaxy than neighboring pixels
        and iteratively refining the grid to find pixels that trace close at higher and higher resolutions. This
        function is core to finding pixelsl that meet this criteria.

        Parameters
        ----------
        lensing_obj
            An object which has a `deflection_2d_from` method for performing lensing calculations, for example a
            `MassProfile`, _Galaxy_, `Plane` or `Tracer`.
        grid : autoarray.Grid2DIrregularUniform or ndarray
            A grid of (y,x) Cartesian coordinates for which the 'peak' values that trace closer to the source than
            their neighbors are found.
        source_plane_coordinate : (y,x)
            The (y,x) coordinate in the source-plane pixels that the distance of traced grid coordinates are computed
            for.
        """

        deflections = deflections_func(grid=grid)
        source_plane_grid = grid.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections
        )

        source_plane_distances = source_plane_grid.distances_to_coordinate_from(
            coordinate=source_plane_coordinate
        )

        neighbors, has_neighbors = grid_square_neighbors_1d_from(
            shape_slim=grid.shape[0]
        )

        grid_peaks = grid_peaks_from(
            distance_1d=source_plane_distances,
            grid_slim=grid,
            neighbors=neighbors.astype("int"),
            has_neighbors=has_neighbors,
        )

        return aa.Grid2DIrregularUniform(
            grid=grid_peaks, pixel_scales=grid.pixel_scales
        )

    def grid_within_distance_of_source_plane_centre(
        self, deflection_func, source_plane_coordinate, grid, distance
    ):
        """
        For an input grid of (y,x) coordinates, remove all coordinates that do not trace within a threshold distance
            of the source-plane centre. This is performed by:

             1) Computing the deflection angle of every (y,x) coordinate on the grid using the input lensing object.
             2) Ray tracing these coordinates to the source-plane.
             3) Computing their distance to the centre of the source in the source-plane.
             4) Removing all coordinates that are not within the input distance of the centre.

            This algorithm is optionally used in the _PositionFiner_. It may be required to remove solutions that are
            genuine 'peaks' that tracer closer to a source than their 8 neighboring pixels, but which do not truly
            trace to the centre of the source-centre.

            Parameters
            ----------
            lensing_obj
                An object which has a `deflection_2d_from` method for performing lensing calculations, for example a
                `MassProfile`, _Galaxy_, `Plane` or `Tracer`.
            grid : autoarray.Grid2DIrregularUniform or ndarray
                A grid of (y,x) Cartesian coordinates for which the 'peak' values that trace closer to the source than
                their neighbors are found.
            source_plane_coordinate : (y,x)
                The (y,x) coordinate in the source-plane pixels that the distance of traced grid coordinates are computed
                for.
            distance
                The distance within which a grid coordinate must trace to the source-plane centre to be retained.
        """
        if distance is None:
            return grid

        deflections = deflection_func(grid=grid)
        source_plane_grid = grid.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections
        )
        source_plane_distances = source_plane_grid.distances_to_coordinate_from(
            coordinate=source_plane_coordinate
        )

        grid_within_distance_of_centre = grid_within_distance(
            distances_1d=source_plane_distances,
            grid_slim=grid,
            within_distance=distance,
        )

        return aa.Grid2DIrregularUniform(
            grid=grid_within_distance_of_centre, pixel_scales=grid.pixel_scales
        )


class PointSolver(AbstractPointSolver):
    def __init__(
        self,
        grid,
        use_upscaling=True,
        pixel_scale_precision=None,
        upscale_factor=2,
        magnification_threshold=0.0,
        distance_to_source_centre=None,
        distance_to_mass_profile_centre=None,
    ):
        """Given a `OperateDeflections` (e.g. a _MassProfile, `Galaxy`, `Plane` or `Tracer`) this class uses their
        deflections_yx_2d_from method to determine the (y,x) coordinates the multiple-images appear given a (y,x)
        source-centre coordinate in the source-plane.

        This is performed as follows:

         1) For an initial input grid, compute all deflection angles, map their values to source-plane and compute the
            distance of each traced coordinate to the source-plane centre.
         2) Find the 'peak' pixels on the image-plane grid. A peak pixel is one that traces closer to the centre of
            the source in the source-plane than it 8 direct neighboring adjacent pixels.
         3) For every peak pixel, create a higher resolution grid around it and centred on it and using this higher
            resolution grid find its peak pixel.

         This process thus finds a set of 'peak' pixels and iteratively refines their values by locating them for
         higher and higher resolution grids. The following occurances may happen during this process:

          - A peak pixel may 'split' into multiple images. This is to be expected, when genuine multiple images
            were previously merged into one due to the grid being too low resolution.

          - Image pixels which do not correspond to genuine multiple images may be detected as they meet the peak
            criteria. This can occurance in certain circumstances where a non-multiple image still traces closer than its
            8 neighbors. Depending on how the `PositionFinder` is being used these can be removed.
        """

        super().__init__(
            use_upscaling=use_upscaling,
            upscale_factor=upscale_factor,
            magnification_threshold=magnification_threshold,
            distance_to_source_centre=distance_to_source_centre,
            distance_to_mass_profile_centre=distance_to_mass_profile_centre,
        )

        self.grid = grid.binned
        self.pixel_scale_precision = pixel_scale_precision

    def refined_coordinates_from(
        self, deflections_func, coordinate, pixel_scale, source_plane_coordinate
    ):
        """
        For an input (y,x) coordinate, determine a set of refined coordinates that are computed by locating peak
        pixels on a higher resolution grid around that pixel.

        This may return 1 or multiple refined coordinates. Multiple coordinates occur when the peak 'splits' into
        multiple images.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate around which the upscaled grid used to fin the refined coordinates is computed.
        pixel_scales : (float, float)
            The pixel-scale resolution of the buffed and upscaled grid that is formed around the input coordinate. If
            upscale > 1, the pixel_scales are reduced to pixel_scale / upscale_factor.
        lensing_obj
            An object which has a `deflection_2d_from` method for performing lensing calculations, for example a
            `MassProfile`, _Galaxy_, `Plane` or `Tracer`.
        source_plane_coordinate : (float, float)
            The (y,x) coordinate in the source-plane pixels that the distance of traced grid coordinates are computed
            for.
        """

        grid = self.grid_buffed_and_upscaled_around_coordinate_from(
            coordinate=coordinate,
            pixel_scales=(pixel_scale, pixel_scale),
            buffer=4,
            upscale_factor=self.upscale_factor,
        )

        grid = self.grid_peaks_from(
            deflections_func=deflections_func,
            grid=grid,
            source_plane_coordinate=source_plane_coordinate,
        )

        if len(grid) == 0:
            return None
        else:
            return [tuple(coordinate) for coordinate in grid]

    def solve(self, lensing_obj, source_plane_coordinate, upper_plane_index=None):

        if upper_plane_index is None:
            deflections_func = lensing_obj.deflections_yx_2d_from
        else:
            deflections_func = partial(
                lensing_obj.deflections_between_planes_from,
                plane_i=0,
                plane_j=upper_plane_index,
            )

        coordinates_list = self.grid_peaks_from(
            grid=self.grid,
            source_plane_coordinate=source_plane_coordinate,
            deflections_func=deflections_func,
        )

        coordinates_list = self.grid_with_coordinates_to_mass_profile_centre_removed_from(
            lensing_obj=lensing_obj, grid=coordinates_list
        )

        coordinates_list = self.grid_with_points_below_magnification_threshold_removed(
            lensing_obj=lensing_obj,
            deflections_func=deflections_func,
            grid=coordinates_list,
        )

        if not self.use_upscaling:

            coordinates_list = self.grid_within_distance_of_source_plane_centre(
                deflection_func=deflections_func,
                grid=aa.Grid2DIrregularUniform(
                    grid=coordinates_list, pixel_scales=self.grid.pixel_scales
                ),
                source_plane_coordinate=source_plane_coordinate,
                distance=self.distance_to_source_centre,
            )

            return aa.Grid2DIrregular(grid=coordinates_list)

        pixel_scale = self.grid.pixel_scale

        while pixel_scale > self.pixel_scale_precision:

            refined_coordinates_list = []

            for coordinate in coordinates_list:

                refined_coordinates = self.refined_coordinates_from(
                    deflections_func=deflections_func,
                    coordinate=coordinate,
                    pixel_scale=pixel_scale,
                    source_plane_coordinate=source_plane_coordinate,
                )

                if refined_coordinates is not None:
                    refined_coordinates_list += refined_coordinates

            refined_coordinates_list = grid_remove_duplicates(
                grid=np.asarray(refined_coordinates_list)
            )

            pixel_scale = pixel_scale / self.upscale_factor

            coordinates_list = refined_coordinates_list

        coordinates_list = self.grid_within_distance_of_source_plane_centre(
            deflection_func=deflections_func,
            grid=aa.Grid2DIrregularUniform(
                grid=coordinates_list, pixel_scales=(pixel_scale, pixel_scale)
            ),
            source_plane_coordinate=source_plane_coordinate,
            distance=self.distance_to_source_centre,
        )

        coordinates_list = self.grid_with_points_below_magnification_threshold_removed(
            lensing_obj=lensing_obj,
            deflections_func=deflections_func,
            grid=coordinates_list,
        )

        return aa.Grid2DIrregular(grid=coordinates_list)


@aa.util.numba.jit()
def grid_remove_duplicates(grid):

    tolerance = 1e-8

    grid_no_duplicates = []

    separations = np.zeros((grid.shape[0], grid.shape[0]))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[0]):
            separations[i, j] = np.sqrt(
                np.square(grid[i, 0] - grid[j, 0]) + np.square(grid[i, 1] - grid[j, 1])
            )
            separations[i, i] = tolerance * 2

    for i in range(grid.shape[0]):

        is_duplicate = False

        for j in range(grid.shape[0]):

            if separations[i, j] < tolerance:

                is_duplicate = True
                separations[i, j] = tolerance * 2
                separations[j, i] = tolerance * 2

        if not is_duplicate:
            grid_no_duplicates.append((grid[i, 0], grid[i, 1]))

    return grid_no_duplicates


@aa.util.numba.jit()
def grid_buffed_around_coordinate_from(
    coordinate, pixel_scales, buffer, upscale_factor=1
):
    """
    From an input 1D grid, return a 1D buffed grid where (y,x) coordinates are added next to all grid pixels whose
    neighbors in the 8 neighboring directions were masked and not included in the grid.

    This is performed by determining the 1D grid's mask in 2D, adding the entries to the 2D mask to the eight
    neighboring pixels and using this buffed mask to create the new 1D buffed grid.

    Parameters
    ----------
    grid_slim
        The irregular 1D grid of (y,x) coordinates over which a square uniform grid is overlaid.
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    total_coordinates = (upscale_factor * (2 * buffer + 1)) ** 2

    grid_slim = np.zeros(shape=(total_coordinates, 2))

    grid_index = 0

    pixel_scales_upscaled = (
        pixel_scales[0] / upscale_factor,
        pixel_scales[1] / upscale_factor,
    )

    y_upscale_half = pixel_scales_upscaled[0] / 2
    x_upscale_half = pixel_scales_upscaled[1] / 2

    edge = int(np.sqrt(total_coordinates))

    if edge % 2 != 0:
        edge_start = -int((edge - 1) / 2)
        edge_end = int((edge - 1) / 2) + 1
        y_odd_pixel_scale = y_upscale_half
        x_odd_pixel_scale = x_upscale_half
    else:
        edge_start = -int(edge / 2)
        edge_end = int(edge / 2)
        y_odd_pixel_scale = 0.0
        x_odd_pixel_scale = 0.0

    for y in range(edge_start, edge_end):
        for x in range(edge_start, edge_end):

            grid_slim[grid_index, 0] = (
                coordinate[0]
                - y * pixel_scales_upscaled[0]
                - y_upscale_half
                + y_odd_pixel_scale
            )
            grid_slim[grid_index, 1] = (
                coordinate[1]
                + x * pixel_scales_upscaled[1]
                + x_upscale_half
                - x_odd_pixel_scale
            )
            grid_index += 1

    return grid_slim


@aa.util.numba.jit()
def pair_coordinate_to_closest_pixel_on_grid(coordinate, grid_slim):

    squared_distances = np.square(grid_slim[:, 0] - coordinate[0]) + np.square(
        grid_slim[:, 1] - coordinate[1]
    )

    return np.argmin(squared_distances)


@aa.util.numba.jit()
def grid_square_neighbors_1d_from(shape_slim):
    """
    From a (y,x) grid of coordinates, determine the 8 neighors of every coordinate on the grid which has 8
    neighboring (y,x) coordinates.

    Neighbor indexes use the 1D index of the pixel on the masked grid counting from the top-left right and down.

    For example:

         x x x  x x x x x x x
         x x x  x x x x x x x      Th s  s an example mask.Mask2D, where:
         x x x  x x x x x x x
         x x x  0 1 2 3 x x x      x = `True` (P xel  s masked and excluded from the gr d)
         x x x  4 5 6 7 x x x      o = `False` (P xel  s not masked and  ncluded  n the gr d)
         x x x  8 9 10 11 x x x
         x x x  x x x x x x x
         x x x  x x x x x x x
         x x x  x x x x x x x
         x x x  x x x x x x x

    On the grid above, the grid cells in 1D indxes 5 and 6 have 8 neighboring pixels and their entries in the
    grid_neighbors_1d array will be:

    grid_neighbors_1d[0,:] = [0, 1, 2, 4, 6, 8, 9, 10]
    grid_neighbors_1d[1,:] = [1, 2, 3, 5, 7, 9, 10, 11]

    The other pixels will be included in the grid_neighbors_1d array, but correspond to `False` entries in
    grid_has_neighbors and be omitted from calculations that use the neighbor array.

    Parameters
    ----------
    shape_slim
        The irregular 1D grid of (y,x) coordinates over which a square uniform grid is overlaid.
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    shape_of_edge = int(np.sqrt(shape_slim))

    has_neighbors = np.full(shape=shape_slim, fill_value=False)
    neighbors_1d = np.full(shape=(shape_slim, 8), fill_value=-1.0)

    index = 0

    for y in range(shape_of_edge):
        for x in range(shape_of_edge):

            if y > 0 and x > 0 and y < shape_of_edge - 1 and x < shape_of_edge - 1:

                neighbors_1d[index, 0] = index - shape_of_edge - 1
                neighbors_1d[index, 1] = index - shape_of_edge
                neighbors_1d[index, 2] = index - shape_of_edge + 1
                neighbors_1d[index, 3] = index - 1
                neighbors_1d[index, 4] = index + 1
                neighbors_1d[index, 5] = index + shape_of_edge - 1
                neighbors_1d[index, 6] = index + shape_of_edge
                neighbors_1d[index, 7] = index + shape_of_edge + 1

                has_neighbors[index] = True

            index += 1

    return neighbors_1d, has_neighbors


@aa.util.numba.jit()
def grid_peaks_from(distance_1d, grid_slim, neighbors, has_neighbors):
    """Given an input grid of (y,x) coordinates and a 1d array of their distances to the centre of the source,
    determine the coordinates which are closer to the source than their 8 neighboring pixels.

    These pixels are selected as the next closest set of pixels to the source and used to define the coordinates of
    the next higher resolution grid.

    Parameters
    ----------
    distance_1d
        The distance of every (y,x) grid coordinate to the centre of the source in the source-plane.
    grid_slim
        The irregular 1D grid of (y,x) coordinates whose distances to the source are compared.
    neighbors
        A 2D array of shape [pixels, 8] giving the 1D index of every grid pixel to its 8 neighboring pixels.
    has_neighbors
        An array of bools, where `True` means a pixel has 8 neighbors and `False` means it has less than 8 and is not
        compared to the source distance.
    """
    peaks_list = []

    for grid_index in range(grid_slim.shape[0]):

        if has_neighbors[grid_index]:

            distance = distance_1d[grid_index]

            if (
                distance <= distance_1d[neighbors[grid_index, 0]]
                and distance <= distance_1d[neighbors[grid_index, 1]]
                and distance <= distance_1d[neighbors[grid_index, 2]]
                and distance <= distance_1d[neighbors[grid_index, 3]]
                and distance <= distance_1d[neighbors[grid_index, 4]]
                and distance <= distance_1d[neighbors[grid_index, 5]]
                and distance <= distance_1d[neighbors[grid_index, 6]]
                and distance <= distance_1d[neighbors[grid_index, 7]]
            ):

                peaks_list.append(grid_slim[grid_index])

    return peaks_list


@aa.util.numba.jit()
def grid_within_distance(distances_1d, grid_slim, within_distance):

    grid_within_size = 0

    for grid_index in range(grid_slim.shape[0]):
        if distances_1d[grid_index] < within_distance:
            grid_within_size += 1

    grid_within = np.zeros(shape=(grid_within_size, 2))

    grid_within_index = 0

    for grid_index in range(grid_slim.shape[0]):
        if distances_1d[grid_index] < within_distance:

            grid_within[grid_within_index, :] = grid_slim[grid_index, :]
            grid_within_index += 1

    return grid_within


@aa.util.numba.jit()
def grid_outside_distance_mask_from(distances_1d, grid_slim, outside_distance):
    grid_outside_size = 0

    for grid_index in range(grid_slim.shape[0]):
        if distances_1d[grid_index] > outside_distance:
            grid_outside_size += 1

    grid_outside = np.zeros(shape=(grid_outside_size, 2))

    grid_outside_index = 0

    for grid_index in range(grid_slim.shape[0]):
        if distances_1d[grid_index] > outside_distance:
            grid_outside[grid_outside_index, :] = grid_slim[grid_index, :]
            grid_outside_index += 1

    return grid_outside
