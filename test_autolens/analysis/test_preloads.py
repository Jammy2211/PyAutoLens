import autofit as af
import autolens as al
from autolens import exc
from autolens.mock import mock
from autolens.analysis import preloads as pload

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
