import numpy as np
from analysis import galaxy
from analysis import ray_tracing
from pixelization import pixelization
from profiling import profiling_data
from profiling import tools

from profiles import mass_profiles


class Reconstructor(object):

    def __init__(self, mapping, regularization, image_to_pix, sub_to_pix):
        """The matrices and mappings used to linearly invert and incorrect_fit a datas-set.

        Parameters
        -----------
        mapping : ndarray
            The matrix representing the mapping_matrix between reconstructed_image-pixels and data_vector-pixels.
        regularization : ndarray
            The matrix defining how the reconstructed_image's pixels are regularized with one another when fitting the
            data_vector.
        image_to_pix : ndarray
            The mapping_matrix between each masked_image-grid pixel and inversion-grid pixel.
        sub_to_pix : ndarray
            The mapping_matrix between each sub-grid pixel and inversion-grid sub-pixel.
        """
        self.mapping = mapping
        self.mapping_shape = mapping.shape
        self.regularization = regularization
        self.image_to_pix = image_to_pix
        self.sub_to_pix = sub_to_pix

    def data_vector_from_blurred_mapping_and_data(self, blurred_mapping, image_vector, noise_vector):
        """ Compute the curvature_matrix matrix directly - used to integration_old test that our curvature_matrix matrix generator approach
        truly works."""
        data_vector = np.zeros(self.mapping_shape[1])

        for image_index in range(self.mapping_shape[0]):
            for pix_index in range(self.mapping_shape[1]):
                data_vector[pix_index] += image_vector[image_index] * \
                                          blurred_mapping[image_index, pix_index] / (noise_vector[image_index] ** 2.0)

        return data_vector


sub_grid_size = 4
psf_size = (41, 41)

sie = mass_profiles.EllipticalIsothermal(centre=(0.010, 0.032), einstein_radius=1.47, axis_ratio=0.849, phi=73.6)
shear = mass_profiles.ExternalShear(magnitude=0.0663, phi=160.5)
lens_galaxy = galaxy.Galaxy(mass_profile_0=sie, mass_profile_1=shear)

source_pix = galaxy.Galaxy(pixelization=pixelization.RectangularRegConst(shape=(19, 19)))

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size, psf_shape=psf_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size, psf_shape=psf_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size, psf_shape=psf_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size, psf_shape=psf_size)
# ao = profiling_data.setup_class(phase_name='AO', pixel_scales=0.01, sub_grid_size=sub_grid_size, psf_shape=psf_size)

lsst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_pix],
                                 image_plane_grids=lsst.grids)
euclid_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_pix],
                                   image_plane_grids=euclid.grids)
hst_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_pix], image_plane_grids=hst.grids)
hst_up_tracer = ray_tracing.Tracer(lens_galaxies=[lens_galaxy], source_galaxies=[source_pix],
                                   image_plane_grids=hst_up.grids)
# ao_tracer = ray_tracing.TracerImagePlane(lens_galaxies=[lens_galaxy], source_galaxies=[source_pix], image_plane_grid=ao.grids)

lsst_recon = lsst_tracer.reconstructors(lsst.borders, cluster_mask=None)
euclid_recon = euclid_tracer.reconstructors(euclid.borders, cluster_mask=None)
hst_recon = hst_tracer.reconstructors(hst.borders, cluster_mask=None)
hst_up_recon = hst_up_tracer.reconstructors(hst_up.borders, cluster_mask=None)
# ao_recon = ao_tracer.reconstructors_from_source_plane(ao.border, cluster_mask=None)

lsst_recon = Reconstructor(lsst_recon.mapping, lsst_recon.regularization, lsst_recon.regular_to_sparse,
                           lsst_recon.sub_to_pix)
euclid_recon = Reconstructor(euclid_recon.mapping, euclid_recon.regularization, euclid_recon.regular_to_sparse,
                             euclid_recon.sub_to_pix)
hst_recon = Reconstructor(hst_recon.mapping, hst_recon.regularization, hst_recon.regular_to_sparse,
                          hst_recon.sub_to_pix)
hst_up_recon = Reconstructor(hst_up_recon.mapping, hst_up_recon.regularization, hst_up_recon.regular_to_sparse,
                             hst_up_recon.sub_to_pix)
# ao_recon = Reconstructor(ao_recon.mapping_matrix, ao_recon.regularization_matrix, ao_recon.image_to_pixelization,
#                            ao_recon.sub_to_pixelization)

lsst_blurred_mapping = lsst.masked_image.convolver_mapping_matrix.convolve_mapping_matrix(lsst_recon.mapping)
euclid_blurred_mapping = euclid.masked_image.convolver_mapping_matrix.convolve_mapping_matrix(euclid_recon.mapping)
hst_blurred_mapping = hst.masked_image.convolver_mapping_matrix.convolve_mapping_matrix(hst_recon.mapping)
hst_up_blurred_mapping = hst_up.masked_image.convolver_mapping_matrix.convolve_mapping_matrix(hst_up_recon.mapping)


# ao_blurred_mapping = ao.masked_image.convolver_mapping_matrix.convolve_mapping_matrix_jit(ao_recon.mapping_matrix)

@tools.tick_toc_x1
def lsst_solution():
    lsst_recon.data_vector_from_blurred_mapping_and_data(lsst_blurred_mapping, lsst.masked_image,
                                                         lsst.masked_image.estimated_noise_map)


@tools.tick_toc_x1
def euclid_solution():
    euclid_recon.data_vector_from_blurred_mapping_and_data(euclid_blurred_mapping, euclid.masked_image,
                                                           euclid.masked_image.estimated_noise_map)


@tools.tick_toc_x1
def hst_solution():
    hst_recon.data_vector_from_blurred_mapping_and_data(hst_blurred_mapping, hst.masked_image,
                                                        hst.masked_image.estimated_noise_map)


@tools.tick_toc_x1
def hst_up_solution():
    hst_up_recon.data_vector_from_blurred_mapping_and_data(hst_up_blurred_mapping, hst_up.masked_image,
                                                           hst_up.masked_image.estimated_noise_map)


@tools.tick_toc_x1
def ao_solution():
    ao_recon.data_vector_from_blurred_mapping_matrix_and_data(ao_blurred_mapping, ao.masked_image.estimated_noise_map)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
