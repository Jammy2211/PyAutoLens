import time

import autolens as al
import numpy as np
from test_autolens.simulators.imaging import instrument_util

repeats = 10

print("Number of repeats = " + str(repeats))
print()

sub_size = 4
radius = 3.6
psf_shape_2d = (11, 11)
pixelization_shape_2d = (20, 20)

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius) + "\n")
print("psf shape = " + str(psf_shape_2d) + "\n")
print("pixelization shape = " + str(pixelization_shape_2d) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

for instrument in ["vro", "euclid", "hst", "hst_up"]:  # , 'ao']:

    imaging = instrument_util.load_test_imaging(
        dataset_name="lens_sie__source_smooth",
        instrument=instrument,
        psf_shape_2d=psf_shape_2d,
    )

    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        sub_size=sub_size,
        radius=radius,
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    print("Rectangular Inversion fit run times for image type " + instrument + "\n")
    print("Number of points = " + str(masked_imaging.grid.sub_shape_1d) + "\n")

    start_overall = time.time()

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    start = time.time()
    for i in range(repeats):
        traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[
            -1
        ]
    diff = time.time() - start
    print("Time to Setup Traced Grid = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
            grid=masked_imaging.grid
        )[-1]

    diff = time.time() - start
    print("Time to Setup Traced Sparse Grid = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        mapper = pixelization.mapper_from_grid_and_sparse_grid(
            grid=traced_grid, sparse_grid=traced_sparse_grid, inversion_use_border=True
        )
    diff = time.time() - start
    print("Time to create mapper (Border Relocation) = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        mapping_matrix = mapper.mapping_matrix
    diff = time.time() - start
    print("Time to compute mapping matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
            mapping_matrix=mapping_matrix
        )
    diff = time.time() - start
    print("Time to compute blurred mapping matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
        )
    diff = time.time() - start
    print("Time to compute data vector = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
        )
    diff = time.time() - start
    print("Time to compute curvature matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            pixel_neighbors=mapper.pixelization_grid.pixel_neighbors,
            pixel_neighbors_size=mapper.pixelization_grid.pixel_neighbors_size,
        )
    diff = time.time() - start
    print("Time to compute reguarization matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
    diff = time.time() - start
    print("Time to compute curvature reguarization Matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
    diff = time.time() - start
    print("Time to peform reconstruction = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        al.util.inversion.mapped_reconstructed_data_from(
            mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
        )
    diff = time.time() - start
    print("Time to compute mapped reconstruction = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
    diff = time.time() - start
    print("Time to perform complete fit = {}".format(diff / repeats))

    print()
