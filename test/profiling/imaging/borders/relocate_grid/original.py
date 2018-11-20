import numpy as np

from profiling import profiling_data
from profiling import tools


class ImageGridBorder(np.ndarray):

    @property
    def no_pixels(self):
        return self.shape[0]

    def __new__(cls, arr, polynomial_degree=3, centre=(0.0, 0.0), *args, **kwargs):
        border = arr.view(cls)
        border.polynomial_degree = polynomial_degree
        border.centre = centre
        return border

    def grid_to_radii(self, grid):
        """
        Convert coordinates to a circular radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid

        Returns
        -------
        The radius at those coordinates
        """

        return np.sqrt(np.add(np.square(np.subtract(grid[:, 0], self.centre[0])),
                              np.square(np.subtract(grid[:, 1], self.centre[1]))))

    def grid_to_thetas(self, grid):
        """
        Compute the angle in degrees between the image_grid and plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        grid : Union((float, float), ndarray)
            The x and y image_grid of the plane.

        Returns
        ----------
        The angle between the image_grid and the x-axis.
        """
        shifted_grid = np.subtract(grid, self.centre)
        theta_from_x = np.degrees(np.arctan2(shifted_grid[:, 1], shifted_grid[:, 0]))
        theta_from_x[theta_from_x < 0.0] += 360.
        return theta_from_x

    def polynomial_fit_to_border(self, grid):

        border_grid = grid[self]

        return np.polyfit(self.grid_to_thetas(border_grid), self.grid_to_radii(border_grid), self.polynomial_degree)

    def move_factors_from_grid(self, grid):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order
         to lie on it. PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying
         they are already within the setup_border_pixels.

        Parameters
        ----------
        grid : ndarray
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        grid_thetas = self.grid_to_thetas(grid)
        grid_radii = self.grid_to_radii(grid)
        poly = self.polynomial_fit_to_border(grid)

        move_factors = np.ones(grid.shape[0])

        for i in range(grid.shape[0]):

            border_radius = np.polyval(poly, grid_thetas[i])

            if grid_radii[i] > border_radius:
                move_factors[i] = border_radius / grid_radii[i]

        return move_factors

    def relocated_grid_from_grid(self, grid):
        move_factors = self.move_factors_from_grid(grid)
        return np.multiply(grid, move_factors[:, None])


class SubGridBorder(ImageGridBorder):

    @classmethod
    def from_mask(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        return cls(mask.edge_sub_pixels(sub_grid_size), polynomial_degree, centre)


sub_grid_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
lsst_border = SubGridBorder.from_mask(mask=lsst.masked_image.mask, sub_grid_size=sub_grid_size)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
euclid_border = SubGridBorder.from_mask(mask=euclid.masked_image.mask, sub_grid_size=sub_grid_size)

hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_border = SubGridBorder.from_mask(mask=hst.masked_image.mask, sub_grid_size=sub_grid_size)

hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
hst_up_border = SubGridBorder.from_mask(mask=hst_up.masked_image.mask, sub_grid_size=sub_grid_size)

ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)
ao_border = SubGridBorder.from_mask(mask=ao.masked_image.mask, sub_grid_size=sub_grid_size)


@tools.tick_toc_x1
def lsst_solution():
    lsst_border.relocated_grid_from_grid(grid=lsst.grids.sub)


@tools.tick_toc_x1
def euclid_solution():
    euclid_border.relocated_grid_from_grid(grid=euclid.grids.sub)


@tools.tick_toc_x1
def hst_solution():
    hst_border.relocated_grid_from_grid(grid=hst.grids.sub)


@tools.tick_toc_x1
def hst_up_solution():
    hst_up_border.relocated_grid_from_grid(grid=hst_up.grids.sub)


@tools.tick_toc_x1
def ao_solution():
    ao_border.relocated_grid_from_grid(grid=ao.grids.sub)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
