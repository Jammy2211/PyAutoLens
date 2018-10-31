import numba
import numpy as np
from analysis import galaxy
from analysis import ray_tracing
from profiling import profiling_data
from profiling import tools

from autolens import exc
from profiles import light_profiles
from profiles import mass_profiles


class RegularizationWeighted(object):
    pixels = None
    regularization_coefficients = None
    pix_signal_scale = None

    def pix_signals_from_images(self, image_to_pix, galaxy_image):
        """Compute the (scaled) signal in each pixel, where the signal is the sum of its masked_image-pixel fluxes. \
        These pix-signals are then used to compute the effective regularization_matrix weight of each pixel.

        The pix signals are scaled in the following ways:

        1) Divided by the number of masked_image-pixels in the pixel, to ensure all pixels have the same \
        'relative' signal (i.e. a pixel with 10 image-pixels doesn't have x2 the signal of one with 5).

        2) Divided by the maximum pix-signal, so that all signals vary between 0 and 1. This ensures that the \
        regularizations weights they're used to compute are defined identically for all masked_image units / SNR's.

        3) Raised to the power of the hyper-parameter *signal_scale*, so the method can control the relative \
        contribution of the different regions of regularization_matrix.
        """

        pix_signals = np.zeros((self.pixels,))
        pix_sizes = np.zeros((self.pixels,))

        for image_index in range(galaxy_image.shape[0]):
            pix_signals[image_to_pix[image_index]] += galaxy_image[image_index]
            pix_sizes[image_to_pix[image_index]] += 1

        pix_signals /= pix_sizes
        pix_signals /= max(pix_signals)

        return pix_signals ** self.pix_signal_scale

    def regularization_weights_from_pix_signals(self, pix_signals):
        """Compute the regularization_matrix weights, which represent the effective regularization_matrix coefficient of every \
        pixel. These are computed using the (scaled) pix-signal in each pixel.

        Two regularization_matrix coefficients are used which mappers to:

        1) pix_signals - This regularizes pix-plane pixels with a high pix-signal (i.e. where the pix is).
        2) 1.0 - pix_signals - This regularizes pix-plane pixels with a low pix-signal (i.e. background sky)
        """
        return (self.regularization_coefficients[0] * pix_signals +
                self.regularization_coefficients[1] * (1.0 - pix_signals)) ** 2.0


class Pixelization(object):

    def __init__(self, pixels=100, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a inversion, which discretizes a set of masked_image and sub grid grid into \
        pixels. These pixels incorrect_fit an masked_image using a linear inversion, where a regularization_matrix matrix
        enforces smoothness between pixel values.

        A number of 1D and 2D arrays are used to represent mappings betwen masked_image, sub, pix, and cluster pixels. The \
        nomenclature here follows grid_to_grid, such that it maps the index of a value on one grid to another. For \
        howtolens:

        - pix_to_image[2] = 5 tells us that the 3rd inversion-pixel maps to the 6th masked_image-pixel.
        - sub_to_pixelization[4,2] = 2 tells us that the 5th sub-pixel maps to the 3rd inversion-pixel.

        Parameters
        ----------
        pixels : int
            The number of pixels in the inversion.
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """
        self.pixels = pixels
        self.regularization_coefficients = regularization_coefficients


class RectangularRegWeight(Pixelization, RegularizationWeighted):

    def __init__(self, shape=(3, 3), regularization_coefficients=(1.0, 1.0), pix_signal_scale=1.0):
        """A rectangular inversion where pixels appear on a Cartesian, uniform and rectangular grid \
        of  shape (rows, columns).

        Like an masked_image grid, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException('The rectangular inversion must be at least dimensions 3x3')

        super(RectangularRegWeight, self).__init__(shape[0] * shape[1], regularization_coefficients)

        self.shape = shape
        self.pix_signal_scale = pix_signal_scale

    class Geometry(object):

        def __init__(self, x_min, x_max, x_pixel_scale, y_min, y_max, y_pixel_scale):
            """The geometry of a rectangular grid, defining where the grids top-left, top-right, bottom-left and \
            bottom-right corners are in arc seconds. The arc-second size of each rectangular pixel is also computed.

            Parameters
            -----------

            """
            self.x_min = x_min
            self.x_max = x_max
            self.x_pixel_scale = x_pixel_scale
            self.y_min = y_min
            self.y_max = y_max
            self.y_pixel_scale = y_pixel_scale

        def arc_second_to_pixel_index_x(self, coordinate):
            return np.floor((coordinate - self.x_min) / self.x_pixel_scale)

        def arc_second_to_pixel_index_y(self, coordinate):
            return np.floor((coordinate - self.y_min) / self.y_pixel_scale)

    def geometry_from_pix_sub_grid(self, pix_sub_grid, buffer=1e-8):
        """Determine the geometry of the rectangular grid, by alligning it with the outer-most pix_grid grid \
        plus a small buffer.

        Parameters
        -----------
        pix_sub_grid : [[float, float]]
            The x and y pix grid (or sub-coordinaates) which are to be matched with their pixels.
        buffer : float
            The size the grid-geometry is extended beyond the most exterior grid.
        """
        x_min = np.min(pix_sub_grid[:, 0]) - buffer
        x_max = np.max(pix_sub_grid[:, 0]) + buffer
        y_min = np.min(pix_sub_grid[:, 1]) - buffer
        y_max = np.max(pix_sub_grid[:, 1]) + buffer
        x_pixel_scale = (x_max - x_min) / self.shape[0]
        y_pixel_scale = (y_max - y_min) / self.shape[1]

        return self.Geometry(x_min, x_max, x_pixel_scale, y_min, y_max, y_pixel_scale)

    def grid_to_pix_from_grid(self, grid, geometry):
        """Compute the mappings between a set of masked_image pixels (or sub-pixels) and pixels, using the masked_image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular inversion's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        grid_to_pix = np.zeros(grid.shape[0], dtype='int')

        for index, pix_coordinate in enumerate(grid):
            x_pixel = geometry.x_arc_seconds_to_pixels(pix_coordinate[0])
            y_pixel = geometry.y_arc_seconds_to_pixels(pix_coordinate[1])

            grid_to_pix[index] = x_pixel * self.shape[1] + y_pixel

        return grid_to_pix

    def grid_to_pix_from_grid_jitted(self, grid, geometry):
        """Compute the mappings between a set of masked_image pixels (or sub-pixels) and pixels, using the masked_image's
        traced pix-plane grid (or sub-grid) and the uniform rectangular inversion's geometry.

        Parameters
        ----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        geometry : Geometry
            The rectangular pixel grid's geometry.
        """
        return self.grid_to_pix_jit(grid, geometry.x_min, geometry.x_pixel_scale, geometry.y_min,
                                    geometry.y_pixel_scale, self.shape[1]).astype(dtype='int')

    @staticmethod
    @numba.jit(nopython=True)
    def grid_to_pix_jit(grid, x_min, x_pixel_scale, y_min, y_pixel_scale, y_shape):

        grid_to_pix = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            x_pixel = np.floor((grid[i, 0] - x_min) / x_pixel_scale)
            y_pixel = np.floor((grid[i, 1] - y_min) / y_pixel_scale)

            grid_to_pix[i] = x_pixel * y_shape + y_pixel

        return grid_to_pix

    def image_to_pix_from_pix_grids(self, grids, borders):
        """
        Compute the inversion matrices of the rectangular inversion by following these steps:

        1) Setup the rectangular grid geometry, by making its corner appear at the higher / lowest x and y pix sub-
        grid.
        2) Pair masked_image and sub-masked_image pixels to the rectangular grid using their traced grid and its geometry.

        Parameters
        ----------

        """
        relocated_grids = borders.relocated_grids_from_grids(grids)
        geometry = self.geometry_from_pix_sub_grid(relocated_grids.sub)
        image_to_pix = self.grid_to_pix_from_grid_jitted(relocated_grids.image_plane_images_, geometry)
        return image_to_pix


sub_grid_size = 4

sie = mass_profiles.EllipticalIsothermal(centre=(0.010, 0.032), einstein_radius=1.47, axis_ratio=0.849, phi=73.6)
shear = mass_profiles.ExternalShear(magnitude=0.0663, phi=160.5)
lens_galaxy = galaxy.Galaxy(mass_profile_0=sie, mass_profile_1=shear)

sersic = light_profiles.EllipticalSersicLightProfile()
source_galaxy = galaxy.Galaxy(light_profile=sersic)

pix = RectangularRegWeight(shape=(50, 50))

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

lsst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                 image_plane_grids=lsst.grids)
euclid_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                   image_plane_grids=euclid.grids)
hst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                image_plane_grids=hst.grids)
hst_up_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                   image_plane_grids=hst_up.grids)
ao_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy], image_plane_grids=ao.grids)

lsst_image = lsst_tracer.source_plane.galaxy_light_profiles_image_from_planes()
euclid_image = lsst_tracer.source_plane.galaxy_light_profiles_image_from_planes()
hst_image = lsst_tracer.source_plane.galaxy_light_profiles_image_from_planes()
hst_up_image = lsst_tracer.source_plane.galaxy_light_profiles_image_from_planes()
ao_image = lsst_tracer.source_plane.galaxy_light_profiles_image_from_planes()

lsst_image_to_pix = pix.image_to_pix_from_pix_grids(grids=lsst_tracer.source_plane.grids, borders=lsst.borders)
euclid_image_to_pix = pix.image_to_pix_from_pix_grids(grids=euclid_tracer.source_plane.grids, borders=euclid.borders)
hst_image_to_pix = pix.image_to_pix_from_pix_grids(grids=hst_tracer.source_plane.grids, borders=hst.borders)
hst_up_image_to_pix = pix.image_to_pix_from_pix_grids(grids=hst_up_tracer.source_plane.grids, borders=hst_up.borders)
ao_image_to_pix = pix.image_to_pix_from_pix_grids(grids=ao_tracer.source_plane.grids, borders=ao.borders)

lsst_pix_signals = pix.pix_signals_from_images(image_to_pix=lsst_image_to_pix, galaxy_image=lsst_image)
euclid_pix_signals = pix.pix_signals_from_images(image_to_pix=euclid_image_to_pix, galaxy_image=euclid_image)
hst_pix_signals = pix.pix_signals_from_images(image_to_pix=hst_image_to_pix, galaxy_image=hst_image)
hst_up_pix_signals = pix.pix_signals_from_images(image_to_pix=hst_up_image_to_pix, galaxy_image=hst_up_image)
ao_pix_signals = pix.pix_signals_from_images(image_to_pix=ao_image_to_pix, galaxy_image=ao_image)


@tools.tick_toc_x1
def lsst_solution():
    pix.regularization_weights_from_pix_signals(lsst_pix_signals)


@tools.tick_toc_x1
def euclid_solution():
    pix.regularization_weights_from_pix_signals(euclid_pix_signals)


@tools.tick_toc_x1
def hst_solution():
    pix.regularization_weights_from_pix_signals(hst_pix_signals)


@tools.tick_toc_x1
def hst_up_solution():
    pix.regularization_weights_from_pix_signals(hst_up_pix_signals)


@tools.tick_toc_x1
def ao_solution():
    pix.regularization_weights_from_pix_signals(ao_pix_signals)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
