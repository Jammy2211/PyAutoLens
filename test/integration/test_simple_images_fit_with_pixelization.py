from autolens.imaging import mask
from autolens.imaging import image
from autolens.imaging import masked_image
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.pixelization import pixelization
from autolens.analysis import fitting
from autolens.analysis import ray_tracing
from autolens.analysis import galaxy

import numpy as np
import pytest


@pytest.fixture(name='sim_grid_9x9', scope='function')
def sim_grid_9x9():
    sim_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(5.5, 5.5), pixel_scale=0.5, psf_size=(3, 3))
    sim_grid_9x9.image_grid = sim_grid_9x9.ma.coordinates_collection_for_subgrid_size_and_blurring_shape(
        sub_grid_size=1,
        blurring_shape=(3, 3))
    sim_grid_9x9.mapping = sim_grid_9x9.ma.grid_mapping_with_sub_grid_size(sub_grid_size=1, cluster_grid_size=1)
    return sim_grid_9x9


@pytest.fixture(name='fit_grid_9x9', scope='function')
def fit_grid_9x9():
    fit_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(4.5, 4.5), pixel_scale=0.5, psf_size=(5, 5))
    fit_grid_9x9.ma = mask.Mask(array=fit_grid_9x9.ma, pixel_scale=1.0)
    fit_grid_9x9.image_grid = mask.GridCollection(fit_grid_9x9.ma, 2, (3, 3,))
    sim_grid_9x9.mapping = sim_grid_9x9.ma.grid_mapping_with_sub_grid_size(sub_grid_size=1, cluster_grid_size=1)
    return fit_grid_9x9


@pytest.fixture(scope='function')
def galaxy_mass_sis():
    sis = mp.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profiles=[sis])


@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=2.0,
                                 sersic_index=1.0)
    return galaxy.Galaxy(light_profiles=[sersic])


class TestCase:

    class TestRectangularPixelization:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)

            ma = mask.Mask.for_simulate(shape_arc_seconds=(3.0, 3.0), pixel_scale=1.0, psf_size=(3, 3))

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]))

            im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((5, 5)))

            mi = masked_image.MaskedImage(im, ma, sub_grid_size=2)

            pix = pixelization.RectangularRegConst(shape=(3, 3), regularization_coefficients=(1.0,))

            galaxy_pix = galaxy.Galaxy(pixelization=pix)

            ray_trace = ray_tracing.Tracer(lens_galaxies=[], source_galaxies=[galaxy_pix], image_plane_grids=mi.grids)

            fitter = fitting.PixelizedFitter(masked_image=mi, sparse_mask=mask.SparseMask(mi.mask, 1), tracer=ray_trace)

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

            reg_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])

            reg_matrix = reg_matrix + 1e-8 * np.identity(9)

            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 0.0008
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = np.log(np.linalg.det(reg_matrix))
            noise_term = 9.0 * np.log(2 * np.pi * 1.0 ** 2.0)

            evidence_expected = -0.5 * (chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitter.fit_data_with_pixelization() == pytest.approx(evidence_expected, 1e-4)

    class TestClusterPixelization:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]).view(image.Image)

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]))

            ma = mask.Mask.for_simulate(shape_arc_seconds=(3.0, 3.0), pixel_scale=1.0, psf_size=(3, 3))

            im = image.Image(im, pixel_scale=1.0, psf=psf, noise=np.ones((5, 5)))

            mi = masked_image.MaskedImage(im, ma, sub_grid_size=2)

            pix = pixelization.ClusterRegConst(pixels=9, regularization_coefficients=(1.0,))

            galaxy_pix = galaxy.Galaxy(pixelization=pix)

            ray_trace = ray_tracing.Tracer(lens_galaxies=[], source_galaxies=[galaxy_pix], image_plane_grids=mi.grids)

            fitter = fitting.PixelizedFitter(masked_image=mi,
                                             sparse_mask=mask.SparseMask(mi.mask, 1), tracer=ray_trace)

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

            reg_matrix = np.array([[2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-1.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 3.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.0, 0.0, 0.0, - 1.0],
                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.0]])

            reg_matrix = reg_matrix + 1e-8 * np.identity(9)

            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 0.0008
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = np.log(np.linalg.det(reg_matrix))
            noise_term = 9.0 * np.log(2 * np.pi * 1.0 ** 2.0)

            evidence_expected = -0.5 * (chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitter.fit_data_with_pixelization() == pytest.approx(evidence_expected, 1e-4)
