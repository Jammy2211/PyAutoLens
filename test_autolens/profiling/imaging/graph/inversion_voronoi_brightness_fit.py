import time

import autolens as al
import numpy as np
import os
import psutil
from test_autolens.simulators.imaging import instrument_util

repeats = 1

print("Number of repeats = " + str(repeats))
print()

sub_size = 4
radius = 3.6
psf_shape_2d = (21, 21)
pixels = 1000

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalPowerLaw(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

pixelization = al.pix.VoronoiBrightnessImage(pixels=pixels)

data_points_total = []
profiling_dict = {}
xticks_list = []
times_list = []
memory_use = []

for instrument in ["vro", "euclid", "hst", "hst_up"]:  # , 'ao']:

    imaging = instrument_util.load_test_imaging(
        data_name="lens_sie__source_smooth", instrument=instrument
    )

    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        sub_size=sub_size,
        radius=radius,
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
        regularization=al.reg.Constant(coefficient=1.0),
        hyper_model_image=masked_imaging.image,
        hyper_galaxy_image=masked_imaging.image,
    )

    data_points_total.append(masked_imaging.grid.mask.pixels_in_mask)

    start_overall = time.time()

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    start = time.time()
    for i in range(repeats):
        traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[
            -1
        ]

    diff = time.time() - start
    profiling_dict["Deflections + \n Subtraction"] = diff / repeats

    start = time.time()
    for i in range(repeats):
        traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
            grid=masked_imaging.grid
        )[-1]
    diff = time.time() - start
    profiling_dict["KMeans"] = diff / repeats

    start = time.time()
    for i in range(repeats):
        mapper = pixelization.mapper_from_grid_and_sparse_grid(
            grid=traced_grid, sparse_grid=traced_sparse_grid, inversion_use_border=True
        )
        mapping_matrix = mapper.mapping_matrix

    diff = time.time() - start
    profiling_dict["Voronoi Mesh + \n Pixel Pairings"] = diff / repeats

    start = time.time()
    for i in range(repeats):
        blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
            mapping_matrix=mapping_matrix
        )
    diff = time.time() - start
    profiling_dict["Mapping Matrix (includes \n PSF convolution)"] = diff / repeats

    start = time.time()
    for i in range(repeats):
        data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
        )
        curvature_matrix = al.util.inversion.curvature_matrix_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            noise_map=masked_imaging.noise_map,
        )
        regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            pixel_neighbors=mapper.pixelization_grid.pixel_neighbors,
            pixel_neighbors_size=mapper.pixelization_grid.pixel_neighbors_size,
        )
        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
        reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
    diff = time.time() - start
    profiling_dict["Linear Algebra"] = diff / repeats

    xticks_list.append(range(len(profiling_dict)))
    times_list.append(list(profiling_dict.values()))

    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1e9
    memory_use.append(memory)

    print(data_points_total)
    print(profiling_dict)

    print(
        profiling_dict["Deflections + \n Subtraction"]
        + profiling_dict["Voronoi Mesh + \n Pixel Pairings"]
        + profiling_dict["Mapping Matrix (includes \n PSF convolution)"]
        + profiling_dict["Linear Algebra"]
    )

print(memory_use)

import pickle

with open(f"data_points_total.pickle", "wb") as f:
    pickle.dump(data_points_total, f)

with open(f"xticks_list.pickle", "wb") as f:
    pickle.dump(xticks_list, f)

with open(f"times_list.pickle", "wb") as f:
    pickle.dump(times_list, f)

with open(f"profiling_dict.pickle", "wb") as f:
    pickle.dump(profiling_dict, f)

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(9, 7))
# plt.plot(np.asarray(xticks_list).T, np.asarray(times_list).T)
#
# xlabels = list(profiling_dict.keys())
# print(xlabels)
# plt.xticks(xticks_list[0], xlabels, size=11, rotation=45)
# plt.ylabel("Run Time (seconds)", fontsize=18)
# plt.legend([f"VRO ({data_points_total[0]} data points)", f"Euclid ({data_points_total[1]} data points)", f"HST ({data_points_total[2]} data points)", f"HST High-res ({data_points_total[3]} data points)"], fontsize=17)
# plt.tight_layout()
# plt.show()
