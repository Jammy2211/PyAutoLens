import sys

sys.path.append("../../")
import numpy as np
from autolens.data.array import mask
from imaging import convolution
import time
import os

path = os.path.dirname(os.path.realpath(__file__))


def load(name):
    return np.load("{}/{}.npy".format(path, name))


grid = load("deflection_data/grid")

psf_shape = (11, 11)

ma = mask.Mask.padded_mask_unmasked_psf_edges(shape_arc_seconds=(4.0, 4.0), pixel_scale=0.1, pad_size=psf_shape)

data = ma.map_2d_array_to_masked_1d_array(np.ones(ma.shape))

mapping = np.ones((len(data), 60))

frame = convolution.Convolver(mask=ma)
convolver = frame.convolver_for_kernel_shape(kernel_shape=psf_shape)
# This PSF leads to no blurring, so equivalent to being off.
kernel_convolver = convolver.convolver_for_kernel(kernel=np.ones(psf_shape))

kernel_convolver.convolve_mapping_matrix(mapping)
repeats = 1


def tick_toc(func):
    def wrapper():
        start = time.time()
        for _ in range(repeats):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))

    return wrapper


@tick_toc
def current_solution():
    kernel_convolver.convolve_mapping_matrix(mapping)


@tick_toc
def jitted_solution():
    kernel_convolver.convolve_mapping_matrix(mapping)


if __name__ == "__main__":
    current_solution()
    jitted_solution()
