import numpy as np
import pytest

import autofit as af

import autolens as al

from autolens.mock.mock import MockMask
from autolens.mock.mock import MockDataset
from autolens.mock.mock import MockTracer
from autolens.mock.mock import MockFit

from autolens import exc

from autolens.lens.model.preloads import Preloads


def test__set_sparse_grid_of_planes():

    # sparse image plane of grids is None so no Preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes is None

    # sparse image plane of grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([2.0])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes is None

    # sparse image plane of grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([1.0])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_grids_of_planes=1)
    preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_grids_of_planes[0] is None
    assert (preloads.sparse_grids_of_planes[1] == np.array([1.0])).all()


# def test__preload_inversion_with_fixed_profiles(fit_imaging_x2_plane_inversion_7x7):
#
#     result = mock.MockResult(
#         max_log_likelihood_fit=fit_imaging_x2_plane_inversion_7x7,
#         max_log_likelihood_pixelization_grids_of_planes=1,
#     )
#
#     # model = af.Collection(
#     #     galaxies=af.Collection(lens=af.Model(al.Galaxy, mass=al.mp.SphIsothermal), source=af.Model(al.Galaxy))
#     # )
#     #
#     # with pytest.raises(exc.PreloadException):
#     #     preload_inversion_with_fixed_profiles(result=result, model=model)
#
#     model = af.Collection(
#         galaxies=af.Collection(
#             lens=af.Model(al.Galaxy),
#             source=af.Model(
#                 al.Galaxy,
#                 pixelization=al.pix.VoronoiBrightnessImage,
#                 regularization=al.reg.AdaptiveBrightness,
#             ),
#         )
#     )
#
#     with pytest.raises(exc.PreloadException):
#         preload_inversion_with_fixed_profiles(result=result, model=model)
#
#     model = af.Collection(
#         galaxies=af.Collection(
#             lens=af.Model(al.Galaxy),
#             source=af.Model(
#                 al.Galaxy,
#                 pixelization=al.pix.VoronoiBrightnessImage(),
#                 regularization=al.reg.AdaptiveBrightness(),
#             ),
#         )
#     )
#
#     preloads = preload_inversion_with_fixed_profiles(result=result, model=model)
#
#     assert preloads.sparse_grids_of_planes == 1
#     assert (
#         preloads.blurred_mapping_matrix
#         == fit_imaging_x2_plane_inversion_7x7.inversion.blurred_mapping_matrix
#     ).all()
#     assert (
#         preloads.curvature_matrix_sparse_preload
#         == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_sparse_preload
#     ).all()
#     assert (
#         preloads.curvature_matrix_preload_counts
#         == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_preload_counts
#     ).all()
#     assert preloads.mapper == fit_imaging_x2_plane_inversion_7x7.inversion.mapper
#
#     preloads = Preloads.setup(result=result, model=model, inversion=True)
#
#     assert preloads.sparse_grids_of_planes == 1
#     assert (
#         preloads.blurred_mapping_matrix
#         == fit_imaging_x2_plane_inversion_7x7.inversion.blurred_mapping_matrix
#     ).all()
#     assert (
#         preloads.curvature_matrix_sparse_preload
#         == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_sparse_preload
#     ).all()
#     assert (
#         preloads.curvature_matrix_preload_counts
#         == fit_imaging_x2_plane_inversion_7x7.inversion.curvature_matrix_preload_counts
#     ).all()
#     assert preloads.mapper == fit_imaging_x2_plane_inversion_7x7.inversion.mapper


def test__set_w_tilde():

    # fit inversion is None, so no need to bother with w_tilde.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fit are different but there is an inversion, so we should not preload w_tilde and use w_tilde.

    fit_0 = MockFit(
        inversion=1, noise_map=al.Array2D.zeros(shape_native=(3, 1), pixel_scales=0.1)
    )
    fit_1 = MockFit(
        inversion=1, noise_map=al.Array2D.ones(shape_native=(3, 1), pixel_scales=0.1)
    )

    preloads = Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fits are the same so preload w_tilde and use it.

    noise_map = al.Array2D.ones(shape_native=(5, 5), pixel_scales=0.1, sub_size=1)

    mask = MockMask(
        _native_index_for_slim_index=noise_map.mask._native_index_for_slim_index
    )

    dataset = MockDataset(psf=al.Kernel2D.no_blur(pixel_scales=1.0), mask=mask)

    fit_0 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)
    fit_1 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)

    preloads = Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde(fit_0=fit_0, fit_1=fit_1)

    curvature_preload, indexes, lengths = al.util.inversion.w_tilde_curvature_preload_imaging_from(
        noise_map_native=fit_0.noise_map.native,
        kernel_native=fit_0.dataset.psf.native,
        native_index_for_slim_index=fit_0.dataset.mask._native_index_for_slim_index,
    )

    assert (preloads.w_tilde.curvature_preload == curvature_preload).all()
    assert (preloads.w_tilde.indexes == indexes).all()
    assert (preloads.w_tilde.lengths == lengths).all()
    assert preloads.w_tilde.noise_map_value == 1.0
    assert preloads.use_w_tilde == True
