from src.imaging import scaled_array
from src.imaging import image
from src.imaging import mask
from src.imaging import grids
from src.profiles import light_profiles as lp
from src.profiles import mass_profiles as mp
from src.pixelization import pixelization
from src.pixelization import frame_convolution
from src.analysis import fitting
from src.analysis import ray_tracing
from src.analysis import galaxy

import numpy as np
import pytest

# TODO : Still suffer border issues described in profile integration test

@pytest.fixture(name='sim_grid_9x9', scope='function')
def sim_grid_9x9():
    sim_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(5.5, 5.5), pixel_scale=0.5, psf_size=(3, 3))
    sim_grid_9x9.image_grid = grids.CoordsCollection.from_mask(mask=sim_grid_9x9.ma, grid_size_sub=1,
                                                               blurring_shape=(3, 3))
    sim_grid_9x9.mappers = grids.MapperCollection.from_mask(mask=sim_grid_9x9.ma)
    return sim_grid_9x9

@pytest.fixture(name='fit_grid_9x9', scope='function')
def fit_grid_9x9():
    fit_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(4.5, 4.5), pixel_scale=0.5, psf_size=(5, 5))
    fit_grid_9x9.ma = mask.Mask(array=fit_grid_9x9.ma, pixel_scale=1.0)
    fit_grid_9x9.image_grid = grids.CoordsCollection.from_mask(mask=fit_grid_9x9.ma, grid_size_sub=2,
                                                               blurring_shape=(3, 3))
    fit_grid_9x9.mappers = grids.MapperCollection.from_mask(mask=fit_grid_9x9.ma)
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
                           [0.0, 0.0, 0.0, 0.0, 0.0]])

            ma = mask.Mask.for_simulate(shape_arc_seconds=(3.0, 3.0), pixel_scale=1.0, psf_size=(3, 3))

            image_grid = grids.CoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_shape=(3, 3))
            grid_datas = grids.DataCollection.from_mask(mask=ma, image=im, noise=np.ones(im.shape),
                                                        exposure_time=np.ones(im.shape))
            mapper_cluster = grids.MapperCluster.from_mask(mask=ma, cluster_grid_size=1)

            ray_trace = ray_tracing.Tracer(lens_galaxies=[], source_galaxies=[], image_plane_grids=image_grid)

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            frame = frame_convolution.FrameMaker(mask=ma)
            convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3))
            # This PSF leads to no blurring, so equivalent to being off.
            kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                               [0., 1., 0.],
                                                                               [0., 0., 0.]]))

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

            reg_matrix = np.array([[ 2.0, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                                   [-1.0,  3.0, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                                   [ 0.0, -1.0,  2.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0],
                                   [-1.0,  0.0,  0.0,  3.0, -1.0,  0.0, -1.0,  0.0,  0.0],
                                   [ 0.0, -1.0,  0.0, -1.0,  4.0, -1.0,  0.0, -1.0,  0.0],
                                   [ 0.0,  0.0, -1.0,  0.0, -1.0,  3.0,  0.0,  0.0,- 1.0],
                                   [ 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  2.0, -1.0,  0.0],
                                   [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  3.0, -1.0],
                                   [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  2.0]])

            reg_matrix = reg_matrix + 1e-8 * np.identity(9)

            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 0.0008
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            det_reg_term = fitting.compute_log_determinant_of_matrix_cholesky(reg_matrix)
            noise_term = 9.0*np.log(2 * np.pi * 1.0 ** 2.0)

            evidence_expected = -0.5*(chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitting.fit_data_with_pixelization(grid_data=grid_datas, pix=pix, kernel_convolver=kernel_convolver,
                                                      tracer=ray_trace, mapper_cluster=mapper_cluster) == \
                   pytest.approx(evidence_expected,1e-4)

    class TestClusterPixelization:

        def test__image_all_1s__direct_image_to_source_mapping__perfect_fit_even_with_regularization(self):

            im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])

            ma = mask.Mask.for_simulate(shape_arc_seconds=(3.0, 3.0), pixel_scale=1.0, psf_size=(3, 3))

            image_grid = grids.CoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_shape=(3, 3))
            grid_datas = grids.DataCollection.from_mask(mask=ma, image=im, noise=np.ones(im.shape),
                                                        exposure_time=np.ones(im.shape))
            mapper_cluster = grids.MapperCluster.from_mask(mask=ma, cluster_grid_size=1)

            ray_trace = ray_tracing.Tracer(lens_galaxies=[], source_galaxies=[], image_plane_grids=image_grid)

            pix = pixelization.ClusterPixelization(pixels=len(mapper_cluster.cluster_to_image),
                                                   regularization_coefficients=(1.0,))

            frame = frame_convolution.FrameMaker(mask=ma)
            convolver = frame.convolver_for_kernel_shape(kernel_shape=(3,3))
            # This PSF leads to no blurring, so equivalent to being off.
            kernel_convolver = convolver.convolver_for_kernel(kernel=np.array([[0., 0., 0.],
                                                                               [0., 1., 0.],
                                                                               [0., 0., 0.]]))

            cov_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

            reg_matrix = np.array([[ 2.0, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                                   [-1.0,  3.0, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                                   [ 0.0, -1.0,  2.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0],
                                   [-1.0,  0.0,  0.0,  3.0, -1.0,  0.0, -1.0,  0.0,  0.0],
                                   [ 0.0, -1.0,  0.0, -1.0,  4.0, -1.0,  0.0, -1.0,  0.0],
                                   [ 0.0,  0.0, -1.0,  0.0, -1.0,  3.0,  0.0,  0.0,- 1.0],
                                   [ 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  2.0, -1.0,  0.0],
                                   [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  3.0, -1.0],
                                   [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  2.0]])

            reg_matrix = reg_matrix + 1e-8 * np.identity(9)

            cov_reg_matrix = cov_matrix + reg_matrix

            chi_sq_term = 0.0
            gl_term = 0.0008
            det_cov_reg_term = np.log(np.linalg.det(cov_reg_matrix))
            print(reg_matrix)
            det_reg_term = fitting.compute_log_determinant_of_matrix_cholesky(reg_matrix)
            noise_term = 9.0*np.log(2 * np.pi * 1.0 ** 2.0)

            evidence_expected = -0.5*(chi_sq_term + gl_term + det_cov_reg_term - det_reg_term + noise_term)

            assert fitting.fit_data_with_pixelization(grid_data=grid_datas, pix=pix, kernel_convolver=kernel_convolver,
                                                      tracer=ray_trace, mapper_cluster=mapper_cluster) == \
                   pytest.approx(evidence_expected,1e-4)