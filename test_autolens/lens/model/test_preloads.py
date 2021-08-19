import numpy as np
import pytest

import autofit as af

import autolens as al

from autolens import exc
from autolens.mock import mock
from autolens.lens.model import preloads as pload


class MockDataset:
    def __init__(self, grid_inversion=None):

        self.grid_inversion = grid_inversion


class MockFit:
    def __init__(self, tracer, dataset=MockDataset()):

        self.dataset = dataset
        self.tracer = tracer


class MockTracer:
    def __init__(self, sparse_image_plane_grids_of_planes=None):

        self.sparse_image_plane_grids_of_planes = sparse_image_plane_grids_of_planes

    def sparse_image_plane_grids_of_planes_from_grid(self, grid):

        return self.sparse_image_plane_grids_of_planes


def test__set_sparse_grid_of_planes():

    # sparse image plane of grids is None so no Preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = pload.Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes is None

    # sparse image plane of grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([2.0])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = pload.Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes is None

    # sparse image plane of grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = pload.Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes[0] is None
    assert (preloads.sparse_grids_of_planes[1] == np.array([1.0])).all()


def test__preload_inversion_with_fixed_profiles(fit_imaging_x2_plane_inversion_7x7):

    result = mock.MockResult(
        max_log_likelihood_fit=fit_imaging_x2_plane_inversion_7x7,
        max_log_likelihood_pixelization_grids_of_planes=1,
    )

    # model = af.Collection(
    #     galaxies=af.Collection(lens=af.Model(al.Galaxy, mass=al.mp.SphIsothermal), source=af.Model(al.Galaxy))
    # )
    #
    # with pytest.raises(exc.PreloadException):
    #     pload.preload_inversion_with_fixed_profiles(result=result, model=model)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy),
            source=af.Model(
                al.Galaxy,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        )
    )

    with pytest.raises(exc.PreloadException):
        pload.preload_inversion_with_fixed_profiles(result=result, model=model)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy),
            source=af.Model(
                al.Galaxy,
                pixelization=al.pix.VoronoiBrightnessImage(),
                regularization=al.reg.AdaptiveBrightness(),
            ),
        )
    )

    preloads = pload.preload_inversion_with_fixed_profiles(result=result, model=model)

    assert preloads.sparse_grids_of_planes == 1
    assert (
        preloads.blurred_mapping_matrix
        == fit_imaging_x2_plane_inversion_7x7.inversion.blurred_mapping_matrix
    ).all()
    assert (
        preloads.curvature_matrix_sparse_preload
        == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_sparse_preload
    ).all()
    assert (
        preloads.curvature_matrix_preload_counts
        == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_preload_counts
    ).all()
    assert preloads.mapper == fit_imaging_x2_plane_inversion_7x7.inversion.mapper

    preloads = pload.Preloads.setup(result=result, model=model, inversion=True)

    assert preloads.sparse_grids_of_planes == 1
    assert (
        preloads.blurred_mapping_matrix
        == fit_imaging_x2_plane_inversion_7x7.inversion.blurred_mapping_matrix
    ).all()
    assert (
        preloads.curvature_matrix_sparse_preload
        == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_sparse_preload
    ).all()
    assert (
        preloads.curvature_matrix_preload_counts
        == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_preload_counts
    ).all()
    assert preloads.mapper == fit_imaging_x2_plane_inversion_7x7.inversion.mapper


def test__preload_w_tilde(fit_imaging_x2_plane_inversion_7x7):

    result = mock.MockResult(
        max_log_likelihood_fit=fit_imaging_x2_plane_inversion_7x7,
        max_log_likelihood_pixelization_grids_of_planes=1,
    )

    # No hyper galaxy is model, so don't need to bother with preloading.

    model = af.Collection(
        galaxies=af.Collection(galaxy=af.Model(al.Galaxy, redshift=0.5))
    )

    preloads = pload.Preloads()
    preloads.set_w_tilde(result=result, model=model)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is None

    # Hyper galaxy is in model, so no preload of w_tilde is used and use_w_tilde=False.

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(al.Galaxy, redshift=0.5, hyper_galaxy=al.HyperGalaxy)
        )
    )

    preloads = pload.Preloads()
    preloads.set_w_tilde(result=result, model=model)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde == False

    # Hyper background noise is in model, so no preload of w_tilde is used and use_w_tilde=False.

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(al.Galaxy, redshift=0.5),
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise,
        )
    )

    preloads = pload.Preloads()
    preloads.set_w_tilde(result=result, model=model)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde == False

    # Hyper galaxy in model but every parameter is fixed (e.g. its an instance) so w_tilde is updated with
    # scaled noise map using this instance.

    hyper_galaxy = af.Model(al.HyperGalaxy)
    hyper_galaxy.contribution_factor = 0.0
    hyper_galaxy.noise_factor = 1.0
    hyper_galaxy.noise_power = 1.0

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(al.Galaxy, redshift=0.5, hyper_galaxy=hyper_galaxy)
        )
    )

    preloads = pload.Preloads()
    preloads.set_w_tilde(result=result, model=model)

    curvature_preload, indexes, lengths = al.util.inversion.w_tilde_curvature_preload_imaging_from(
        noise_map_native=result.max_log_likelihood_fit.noise_map.native,
        kernel_native=result.max_log_likelihood_fit.dataset.psf.native,
        native_index_for_slim_index=result.max_log_likelihood_fit.mask._native_index_for_slim_index,
    )

    assert (preloads.w_tilde.curvature_preload == curvature_preload).all()
    assert (preloads.w_tilde.indexes == indexes).all()
    assert (preloads.w_tilde.lengths == lengths).all()
    assert preloads.w_tilde.noise_map_value == 2.0
    assert preloads.use_w_tilde == True

    # Hyper background is in model but every parameter is fixed (e.g. its an instance) so w_tilde is updated with
    # scaled noise map using this instance.

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(al.Galaxy, redshift=0.5),
            hyper_background_noise=al.hyper_data.HyperBackgroundNoise(noise_scale=1.0),
        )
    )

    preloads = pload.Preloads()
    preloads.set_w_tilde(result=result, model=model)

    curvature_preload, indexes, lengths = al.util.inversion.w_tilde_curvature_preload_imaging_from(
        noise_map_native=result.max_log_likelihood_fit.noise_map.native,
        kernel_native=result.max_log_likelihood_fit.dataset.psf.native,
        native_index_for_slim_index=result.max_log_likelihood_fit.mask._native_index_for_slim_index,
    )

    assert (preloads.w_tilde.curvature_preload == curvature_preload).all()
    assert (preloads.w_tilde.indexes == indexes).all()
    assert (preloads.w_tilde.lengths == lengths).all()
    assert preloads.w_tilde.noise_map_value == 2.0
    assert preloads.use_w_tilde == True
