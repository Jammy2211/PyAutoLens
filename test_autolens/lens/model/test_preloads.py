import autofit as af
import autolens as al
from autolens import exc
from autolens.mock import mock
from autolens.lens.model import preloads as pload

import pytest


def test__preload_pixelization():

    result = mock.MockResult(max_log_likelihood_pixelization_grids_of_planes=1)

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(al.Galaxy), source=af.Model(al.Galaxy))
    )

    with pytest.raises(exc.PreloadException):
        pload.preload_pixelization_grid_from(result=result, model=model)

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
        pload.preload_pixelization_grid_from(result=result, model=model)

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

    preloads = pload.preload_pixelization_grid_from(result=result, model=model)

    assert preloads.sparse_grids_of_planes == 1

    preloads = pload.Preloads.setup(result=result, model=model, pixelization=True)

    assert preloads.sparse_grids_of_planes == 1


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

    preloads = pload.Preloads.setup(result=result, model=model, w_tilde=True)

    assert preloads.w_tilde == None
    assert preloads.use_w_tilde == None

    # Hyper galaxy is in model, so no preload of w_tilde is used and use_w_tilde=False.

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(al.Galaxy, redshift=0.5, hyper_galaxy=al.HyperGalaxy)
        )
    )

    preloads = pload.Preloads.setup(result=result, model=model, w_tilde=True)

    assert preloads.w_tilde == None
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

    preloads = pload.Preloads.setup(result=result, model=model, w_tilde=True)

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
