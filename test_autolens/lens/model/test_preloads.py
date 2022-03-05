import numpy as np
from os import path

import autofit as af

import autolens as al


def test__set_blurred_image():

    # Blurred image is all zeros so preloads as zeros

    fit_0 = al.m.MockFitImaging(blurred_image=np.zeros(2))
    fit_1 = al.m.MockFitImaging(blurred_image=np.zeros(2))

    preloads = al.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.zeros(2)).all()

    # Blurred image are different, indicating the model parameters change the grid, so no preloading.

    fit_0 = al.m.MockFitImaging(blurred_image=np.array([1.0]))
    fit_1 = al.m.MockFitImaging(blurred_image=np.array([2.0]))

    preloads = al.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert preloads.blurred_image is None

    # Blurred images are the same meaning they are fixed in the model, so do preload.

    fit_0 = al.m.MockFitImaging(blurred_image=np.array([1.0]))
    fit_1 = al.m.MockFitImaging(blurred_image=np.array([1.0]))

    preloads = al.Preloads(blurred_image=1)
    preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.blurred_image == np.array([1.0])).all()


def test__set_traced_grids_of_planes():

    # traced grids is None so no Preloading.

    tracer_0 = al.m.MockTracer(traced_grids_of_planes=[None, None])
    tracer_1 = al.m.MockTracer(traced_grids_of_planes=[None, None])

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion is None

    # traced grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = al.m.MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = al.m.MockTracer(traced_grids_of_planes=[None, np.array([[2.0]])])

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion is None

    # traced grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = al.m.MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])
    tracer_1 = al.m.MockTracer(traced_grids_of_planes=[None, np.array([[1.0]])])

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(traced_grids_of_planes_for_inversion=1)
    preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)

    assert preloads.traced_grids_of_planes_for_inversion[0] is None
    assert (preloads.traced_grids_of_planes_for_inversion[1] == np.array([[1.0]])).all()


def test__set_sparse_grid_of_planes():

    # sparse image plane of grids is None so no Preloading.

    tracer_0 = al.m.MockTracer(sparse_image_plane_grid_pg_list=[None, None])
    tracer_1 = al.m.MockTracer(sparse_image_plane_grid_pg_list=[None, None])

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(sparse_image_plane_grid_pg_list=1)
    preloads.set_sparse_image_plane_grid_pg_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grid_pg_list is None

    # sparse image plane of grids are different, indiciating the model parameters change the grid, so no preloading.

    tracer_0 = al.m.MockTracer(
        sparse_image_plane_grid_pg_list=[None, np.array([[1.0]])]
    )
    tracer_1 = al.m.MockTracer(
        sparse_image_plane_grid_pg_list=[None, np.array([[2.0]])]
    )

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(sparse_image_plane_grid_pg_list=1)
    preloads.set_sparse_image_plane_grid_pg_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grid_pg_list is None

    # sparse image plane of grids are the same meaning they are fixed in the model, so do preload.

    tracer_0 = al.m.MockTracer(
        sparse_image_plane_grid_pg_list=[None, np.array([[1.0]])]
    )
    tracer_1 = al.m.MockTracer(
        sparse_image_plane_grid_pg_list=[None, np.array([[1.0]])]
    )

    fit_0 = al.m.MockFitImaging(tracer=tracer_0)
    fit_1 = al.m.MockFitImaging(tracer=tracer_1)

    preloads = al.Preloads(sparse_image_plane_grid_pg_list=1)
    preloads.set_sparse_image_plane_grid_pg_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.sparse_image_plane_grid_pg_list[0] is None
    assert (preloads.sparse_image_plane_grid_pg_list[1] == np.array([[1.0]])).all()


def test__info():

    file_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

    file_preloads = path.join(file_path, "preloads.summary")

    preloads = al.Preloads(
        blurred_image=np.zeros(3),
        w_tilde=None,
        use_w_tilde=False,
        traced_grids_of_planes_for_inversion=None,
        sparse_image_plane_grid_pg_list=None,
        relocated_grid=None,
        linear_obj_list=None,
        operated_mapping_matrix=None,
        curvature_matrix_preload=None,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    i = 0

    assert lines[i] == f"W Tilde = False\n"
    i += 1
    assert lines[i] == f"Use W Tilde = False\n"
    i += 1
    assert lines[i] == f"\n"
    i += 1
    assert lines[i] == f"Blurred Image = False\n"
    i += 1
    assert lines[i] == f"Traced Grids of Planes (For LEq) = False\n"
    i += 1
    assert lines[i] == f"Sparse Image-Plane Grids of Planes = False\n"
    i += 1
    assert lines[i] == f"Relocated Grid = False\n"
    i += 1
    assert lines[i] == f"Mapper = False\n"
    i += 1
    assert lines[i] == f"Blurred Mapping Matrix = False\n"
    i += 1
    assert lines[i] == f"Curvature Matrix Sparse = False\n"
    i += 1
    assert lines[i] == f"Regularization Matrix = False\n"
    i += 1
    assert lines[i] == f"Log Det Regularization Matrix Term = False\n"
    i += 1

    preloads = al.Preloads(
        blurred_image=1,
        w_tilde=1,
        use_w_tilde=True,
        traced_grids_of_planes_for_inversion=1,
        relocated_grid=1,
        sparse_image_plane_grid_pg_list=1,
        linear_obj_list=1,
        operated_mapping_matrix=1,
        curvature_matrix_preload=1,
        regularization_matrix=1,
        log_det_regularization_matrix_term=1,
    )

    af.formatter.output_list_of_strings_to_file(
        file=file_preloads, list_of_strings=preloads.info
    )

    results = open(file_preloads)
    lines = results.readlines()

    i = 0

    assert lines[i] == f"W Tilde = True\n"
    i += 1
    assert lines[i] == f"Use W Tilde = True\n"
    i += 1
    assert lines[i] == f"\n"
    i += 1
    assert lines[i] == f"Blurred Image = True\n"
    i += 1
    assert lines[i] == f"Traced Grids of Planes (For LEq) = True\n"
    i += 1
    assert lines[i] == f"Sparse Image-Plane Grids of Planes = True\n"
    i += 1
    assert lines[i] == f"Relocated Grid = True\n"
    i += 1
    assert lines[i] == f"Mapper = True\n"
    i += 1
    assert lines[i] == f"Blurred Mapping Matrix = True\n"
    i += 1
    assert lines[i] == f"Curvature Matrix Sparse = True\n"
    i += 1
    assert lines[i] == f"Regularization Matrix = True\n"
    i += 1
    assert lines[i] == f"Log Det Regularization Matrix Term = True\n"
    i += 1
