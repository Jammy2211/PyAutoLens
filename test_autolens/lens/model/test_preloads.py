import numpy as np
from os import path

import autofit as af

from autoarray.mock.mock import MockMapper
from autoarray.mock.mock import MockInversion

import autolens as al

from autolens.mock.mock import MockMask
from autolens.mock.mock import MockDataset
from autolens.mock.mock import MockTracer
from autolens.mock.mock import MockFit
from autolens.lens.model.preloads import Preloads


def test__set_blurred_image():

    # Blurred image is all zeros so preloads as zeros

    fit_0 = MockFit(blurred_image=np.zeros(2))
    fit_1 = MockFit(blurred_image=np.zeros(2))

    preloads = Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.zeros(2)).all()

    # Blurred image are different, indicating the model parameters change the grid, so no preloading.

    fit_0 = MockFit(blurred_image=np.array([1.0]))
    fit_1 = MockFit(blurred_image=np.array([2.0]))

    preloads = Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_image is None

    # Blurred images are the same meaning they are fixed in the model, so do preload.

    fit_0 = MockFit(blurred_image=np.array([1.0]))
    fit_1 = MockFit(blurred_image=np.array([1.0]))

    preloads = Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.array([1.0])).all()


def test__set_w_tilde():

    # fit inversion is None, so no need to bother with w_tilde.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

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
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fits are the same so preload w_tilde and use it.

    noise_map = al.Array2D.ones(shape_native=(5, 5), pixel_scales=0.1, sub_size=1)

    mask = MockMask(
        native_index_for_slim_index=noise_map.mask.native_index_for_slim_index
    )

    dataset = MockDataset(psf=al.Kernel2D.no_blur(pixel_scales=1.0), mask=mask)

    fit_0 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)
    fit_1 = MockFit(inversion=1, dataset=dataset, noise_map=noise_map)

    preloads = Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    curvature_preload, indexes, lengths = al.util.inversion.w_tilde_curvature_preload_imaging_from(
        noise_map_native=fit_0.noise_map.native,
        signal_to_noise_map_native=fit_0.signal_to_noise_map.native,
        kernel_native=fit_0.dataset.psf.native,
        native_index_for_slim_index=fit_0.dataset.mask.native_index_for_slim_index,
    )

    assert (preloads.w_tilde.curvature_preload == curvature_preload).all()
    assert (preloads.w_tilde.indexes == indexes).all()
    assert (preloads.w_tilde.lengths == lengths).all()
    assert preloads.w_tilde.noise_map_value == 1.0
    assert preloads.use_w_tilde == True


def test__set_traced_grids_of_planes():

    # traced grids is None so no Preloading.

    tracer_0 = MockTracer(traced_grids_of_planes=[None, None])
    tracer_1 = MockTracer(traced_grids_of_planes=[None, None])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion is None

    # traced grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = MockTracer(traced_grids_of_planes=[None, np.array([[2.0]])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion is None

    # traced grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion[0] is None
    assert (preloads.traced_grids_of_planes_for_inversion[1] == np.array([[1.0]])).all()


def test__set_relocated_grid():

    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))
    inversion_1 = MockInversion(
        mapper=MockMapper(source_grid_slim=2.0 * np.ones((3, 2)))
    )

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(source_grid_slim=np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.relocated_grid == np.ones((3, 2))).all()


def test__set_sparse_grid_of_planes():

    # sparse image plane of grids is None so no Preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, None])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_image_plane_grids_of_planes=1)
    preloads.set_sparse_image_plane_grids_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grids_of_planes is None

    # sparse image plane of grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([[2.0]])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_image_plane_grids_of_planes=1)
    preloads.set_sparse_image_plane_grids_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grids_of_planes is None

    # sparse image plane of grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = MockTracer(sparse_image_plane_grids_of_planes=[None, np.array([[1.0]])])

    fit_0 = MockFit(tracer=tracer_0)
    fit_1 = MockFit(tracer=tracer_1)

    preloads = Preloads(sparse_image_plane_grids_of_planes=1)
    preloads.set_sparse_image_plane_grids_of_planes(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grids_of_planes[0] is None
    assert (preloads.sparse_image_plane_grids_of_planes[1] == np.array([[1.0]])).all()


def test__set_mapper():

    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(mapping_matrix=2.0 * np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))
    inversion_1 = MockInversion(mapper=MockMapper(mapping_matrix=np.ones((3, 2))))

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(mapper=1)
    preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper.mapping_matrix == np.ones((3, 2))).all()


def test__set_inversion():

    curvature_matrix_sparse_preload = np.array([[1.0]])
    curvature_matrix_preload_counts = np.array([1.0])

    # Inversion is None thus preload it to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=np.array([[1.0]]),
        curvature_matrix_preload_counts=np.array([1.0]),
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_mapping_matrix is None
    assert preloads.curvature_matrix_sparse_preload is None
    assert preloads.curvature_matrix_preload_counts is None

    # Inversion's blurred mapping matrices are different thus no preloading.

    blurred_mapping_matrix_0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    blurred_mapping_matrix_1 = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    inversion_0 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_0)
    inversion_1 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_mapping_matrix is None
    assert preloads.curvature_matrix_sparse_preload is None
    assert preloads.curvature_matrix_preload_counts is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = MockInversion(
        blurred_mapping_matrix=blurred_mapping_matrix_0,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    inversion_1 = MockInversion(blurred_mapping_matrix=blurred_mapping_matrix_0)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=curvature_matrix_preload_counts,
    )
    preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_mapping_matrix == blurred_mapping_matrix_0).all()
    assert (
        preloads.curvature_matrix_sparse_preload
        == curvature_matrix_sparse_preload.astype("int")
    ).all()
    assert (
        preloads.curvature_matrix_preload_counts
        == curvature_matrix_preload_counts.astype("int")
    ).all()


def test__set_regularization_matrix_and_term():

    # Inversion is None thus preload log_det_regularization_matrix_term to None.

    fit_0 = MockFit(inversion=None)
    fit_1 = MockFit(inversion=None)

    preloads = Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's log_det_regularization_matrix_term are different thus no preloading.

    inversion_0 = MockInversion(log_det_regularization_matrix_term=0)
    inversion_1 = MockInversion(log_det_regularization_matrix_term=1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = MockInversion(log_det_regularization_matrix_term=1)
    inversion_1 = MockInversion(log_det_regularization_matrix_term=1)

    fit_0 = MockFit(inversion=inversion_0)
    fit_1 = MockFit(inversion=inversion_1)

    preloads = Preloads(log_det_regularization_matrix_term=2)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.regularization_matrix == np.zeros((1, 1))).all()
    assert preloads.log_det_regularization_matrix_term == 1


def test__info():

    file_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

    file_preloads = path.join(file_path, "preloads.summary")

    preloads = Preloads(
        blurred_image=np.zeros(3),
        w_tilde=None,
        use_w_tilde=False,
        traced_grids_of_planes_for_inversion=None,
        sparse_image_plane_grids_of_planes=None,
        relocated_grid=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    assert lines[0] == f"Blurred Image = False\n"
    assert lines[1] == f"W Tilde = False\n"
    assert lines[2] == f"Use W Tilde = False\n"
    assert lines[3] == f"Traced Grids of Planes (For Inversion) = False\n"
    assert lines[4] == f"Sparse Image-Plane Grids of Planes = False\n"
    assert lines[5] == f"Relocated Grid = False\n"
    assert lines[6] == f"Mapper = False\n"
    assert lines[7] == f"Blurred Mapping Matrix = False\n"
    assert lines[8] == f"Curvature Matrix Sparse = False\n"
    assert lines[9] == f"Regularization Matrix = False\n"
    assert lines[10] == f"Log Det Regularization Matrix Term = False\n"

    preloads = Preloads(
        blurred_image=1,
        w_tilde=1,
        use_w_tilde=True,
        traced_grids_of_planes_for_inversion=1,
        relocated_grid=1,
        sparse_image_plane_grids_of_planes=1,
        mapper=1,
        blurred_mapping_matrix=1,
        curvature_matrix_sparse_preload=1,
        regularization_matrix=1,
        log_det_regularization_matrix_term=1,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    assert lines[0] == f"Blurred Image = True\n"
    assert lines[1] == f"W Tilde = True\n"
    assert lines[2] == f"Use W Tilde = True\n"
    assert lines[3] == f"Traced Grids of Planes (For Inversion) = True\n"
    assert lines[4] == f"Sparse Image-Plane Grids of Planes = True\n"
    assert lines[5] == f"Relocated Grid = True\n"
    assert lines[6] == f"Mapper = True\n"
    assert lines[7] == f"Blurred Mapping Matrix = True\n"
    assert lines[8] == f"Curvature Matrix Sparse = True\n"
    assert lines[9] == f"Regularization Matrix = True\n"
    assert lines[10] == f"Log Det Regularization Matrix Term = True\n"
