import scipy
from analysis import galaxy
from analysis import ray_tracing
from profiling import profiling_data
from profiling import tools

from profiles import mass_profiles


class Voronoi(object):

    def __init__(self, pixels=100, regularization_coefficients=(1.0,)):
        """
        Abstract base class for a Voronoi inversion, which represents pixels as a set of centers where \
        all of the nearest-neighbor pix-grid (i.e. traced masked_image-pixels) are mapped to them.

        This forms a Voronoi grid pix-plane, the properties of which are used for fast calculations, defining the \
        regularization_matrix matrix and visualization.

        Parameters
        ----------
        pixels : int
            The number of pixels in the inversion.
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """

        super(Voronoi, self).__init__(pixels, regularization_coefficients)

    @staticmethod
    def voronoi_from_cluster_grid(cluster_grid):
        """Compute the Voronoi grid of the inversion, using the pixel centers.

        Parameters
        ----------
        cluster_grid : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(cluster_grid, qhull_options='Qbb Qc Qx Qm')


sub_grid_size = 4

pix = Voronoi(pixels=200)

sie = mass_profiles.EllipticalIsothermal(centre=(0.010, 0.032), einstein_radius=1.47, axis_ratio=0.849, phi=73.6)
shear = mass_profiles.ExternalShear(magnitude=0.0663, phi=160.5)

lens_galaxy = galaxy.Galaxy(mass_profile_0=sie, mass_profile_1=shear)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

lsst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=lsst.grids)
euclid_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=euclid.grids)
hst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=hst.grids)
hst_up_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=hst_up.grids)
ao_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[], image_plane_grids=ao.grids)


@tools.tick_toc_x1
def lsst_solution():
    pix.voronoi_from_cluster_grid()


@tools.tick_toc_x1
def euclid_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=euclid_sub_to_pix, grids=euclid.grids)


@tools.tick_toc_x1
def hst_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=hst_sub_to_pix, grids=hst.grids)


@tools.tick_toc_x1
def hst_up_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=hst_up_sub_to_pix, grids=hst_up.grids)


@tools.tick_toc_x1
def ao_solution():
    pix.mapping_matrix_from_sub_to_pix(sub_to_pix=ao_sub_to_pix, grids=ao.grids)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
