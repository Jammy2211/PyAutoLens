import time

from autolens.imaging import mask

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
def lsst_current_solution():
    mask.Mask.unmasked_for_shape_and_pixel_scale(shape=(10.0, 10.0), pixel_scale=0.2)


@tick_toc
def euclid_current_solution():
    mask.Mask.unmasked_for_shape_and_pixel_scale(shape=(10.0, 10.0), pixel_scale=0.1)


@tick_toc
def hst_current_solution():
    mask.Mask.unmasked_for_shape_and_pixel_scale(shape=(10.0, 10.0), pixel_scale=0.05)


@tick_toc
def hst_up_current_solution():
    mask.Mask.unmasked_for_shape_and_pixel_scale(shape=(10.0, 10.0), pixel_scale=0.03)


@tick_toc
def ao_current_solution():
    mask.Mask.unmasked_for_shape_and_pixel_scale(shape=(10.0, 10.0), pixel_scale=0.01)


if __name__ == "__main__":
    lsst_current_solution()
    euclid_current_solution()
    hst_current_solution()
    hst_up_current_solution()
    ao_current_solution()
