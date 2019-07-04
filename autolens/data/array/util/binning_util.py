from autolens import decorator_util
import numpy as np

from autolens.data.array.util import array_util, mask_util


@decorator_util.jit()
def padded_binning_shape_2d_from_shape_2d_and_bin_up_factor(shape_2d, bin_up_factor):

    shape_remainder = (shape_2d[0] % bin_up_factor, shape_2d[1] % bin_up_factor)

    if shape_remainder[0] != 0 and shape_remainder[1] != 0:
        shape_pad = (bin_up_factor - shape_remainder[0], bin_up_factor - shape_remainder[1])
    elif shape_remainder[0] != 0 and shape_remainder[1] == 0:
        shape_pad = (bin_up_factor - shape_remainder[0], 0)
    elif shape_remainder[0] == 0 and shape_remainder[1] != 0:
        shape_pad = (0, bin_up_factor - shape_remainder[1])
    else:
        shape_pad = (0, 0)

    return (shape_2d[0] + shape_pad[0], shape_2d[1] + shape_pad[1])

@decorator_util.jit()
def padded_binning_array_2d_from_array_2d_and_bin_up_factor(array_2d, bin_up_factor, pad_value=0.0):
    """If an array is to be binned up, but the dimensions are not divisible by the bin-up factor, this routine pads \
    the array to make it divisible.

    For example, if the array is shape (5,5) and the bin_up_factor is 2, this routine will pad the array to shape \
    (6,6).

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is padded.
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).
    pad_value : float
        If the array is padded, the value the padded edge values are filled in using.

    Returns
    -------
    ndarray
        The 2D array that is padded before binning up.

    Examples
    --------
    array_2d = np.ones((5,5))
    padded_array_2d = padded_array_2d_for_binning_up_with_bin_up_factor( \
        array_2d=array_2d, bin_up_factor=2, pad_value=0.0)
    """

    padded_binning_shape_2d = padded_binning_shape_2d_from_shape_2d_and_bin_up_factor(
        shape_2d=array_2d.shape, bin_up_factor=bin_up_factor)

    return array_util.resized_array_2d_from_array_2d_and_resized_shape(
        array_2d=array_2d, resized_shape=padded_binning_shape_2d, pad_value=pad_value)

@decorator_util.jit()
def binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(array_2d, bin_up_factor):
    """Bin up an array to coarser resolution, by binning up groups of pixels and using their mean value to determine \
     the value of the new pixel.

    If an array of shape (8,8) is input and the bin up size is 2, this would return a new array of size (4,4) where \
    every pixel was the mean of each collection of 2x2 pixels on the (8,8) array.

    If binning up the array leads to an edge being cut (e.g. a (9,9) array binned up by 2), the array is first \
    padded to make the division work. One must be careful of edge effects in this case.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is binned up.
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The binned up 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=2)
    """

    padded_binning_array_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=array_2d, bin_up_factor=bin_up_factor)

    binned_array_2d = np.zeros(shape=(padded_binning_array_2d.shape[0] // bin_up_factor,
                                      padded_binning_array_2d.shape[1] // bin_up_factor))

    for y in range(binned_array_2d.shape[0]):
        for x in range(binned_array_2d.shape[1]):
            value = 0.0
            for y1 in range(bin_up_factor):
                for x1 in range(bin_up_factor):
                    padded_y = y*bin_up_factor + y1
                    padded_x = x*bin_up_factor + x1
                    value += padded_binning_array_2d[padded_y, padded_x]

            binned_array_2d[y,x] = value / (bin_up_factor ** 2.0)

    return binned_array_2d

@decorator_util.jit()
def binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(array_2d, bin_up_factor):
    """Bin up an array to coarser resolution, by binning up groups of pixels and using their quadrature value to \
    determine the value of the new pixel.

    If an array of shape (8,8) is input and the bin up size is 2, this would return a new array of size (4,4) where \
    every pixel was the quadrature of each collection of 2x2 pixels on the (8,8) array.

    If binning up the array leads to an edge being cut (e.g. a (9,9) array binned up by 2), the array is first \
    padded to make the division work. One must be careful of edge effects in this case.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is binned up.
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The binned up 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=2)
    """

    padded_binning_array_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=array_2d, bin_up_factor=bin_up_factor)

    binned_array_2d = np.zeros(shape=(padded_binning_array_2d.shape[0] // bin_up_factor,
                                      padded_binning_array_2d.shape[1] // bin_up_factor))

    for y in range(binned_array_2d.shape[0]):
        for x in range(binned_array_2d.shape[1]):
            value = 0.0
            for y1 in range(bin_up_factor):
                for x1 in range(bin_up_factor):
                    padded_y = y*bin_up_factor + y1
                    padded_x = x*bin_up_factor + x1
                    value += padded_binning_array_2d[padded_y, padded_x] ** 2.0

            binned_array_2d[y,x] = np.sqrt(value) / (bin_up_factor ** 2.0)

    return binned_array_2d

@decorator_util.jit()
def binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(array_2d, bin_up_factor):
    """Bin up an array to coarser resolution, by binning up groups of pixels and using their sum value to determine \
     the value of the new pixel.

    If an array of shape (8,8) is input and the bin up size is 2, this would return a new array of size (4,4) where \
    every pixel was the sum of each collection of 2x2 pixels on the (8,8) array.

    If binning up the array leads to an edge being cut (e.g. a (9,9) array binned up by 2), the array is first \
    padded to make the division work. One must be careful of edge effects in this case.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is binned up.
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The binned up 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=2)
    """

    padded_binning_array_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=array_2d, bin_up_factor=bin_up_factor)

    binned_array_2d = np.zeros(shape=(padded_binning_array_2d.shape[0] // bin_up_factor,
                                      padded_binning_array_2d.shape[1] // bin_up_factor))

    for y in range(binned_array_2d.shape[0]):
        for x in range(binned_array_2d.shape[1]):
            value = 0.0
            for y1 in range(bin_up_factor):
                for x1 in range(bin_up_factor):
                    padded_y = y*bin_up_factor + y1
                    padded_x = x*bin_up_factor + x1
                    value += padded_binning_array_2d[padded_y, padded_x]

            binned_array_2d[y,x] = value

    return binned_array_2d

@decorator_util.jit()
def binned_up_mask_2d_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Bin up an array to coarser resolution, by binning up groups of pixels and using their sum value to determine \
     the value of the new pixel.

    If an array of shape (8,8) is input and the bin up size is 2, this would return a new array of size (4,4) where \
    every pixel was the sum of each collection of 2x2 pixels on the (8,8) array.

    If binning up the array leads to an edge being cut (e.g. a (9,9) array binned up by 2), an array is first \
    extracted around the centre of that array.


    Parameters
    ----------
    mask_2d : ndarray
        The 2D array that is resized.
    new_shape : (int, int)
        The (y,x) new pixel dimension of the trimmed array.
    origin : (int, int)
        The oigin of the resized array, e.g. the central pixel around which the array is extracted.

    Returns
    -------
    ndarray
        The resized 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = resize_array_2d(array_2d=array_2d, new_shape=(2,2), origin=(2, 2))
    """

    padded_mask_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=mask_2d, bin_up_factor=bin_up_factor, pad_value=True)

    binned_mask_2d = np.zeros(shape=(padded_mask_2d.shape[0] // bin_up_factor,
                                      padded_mask_2d.shape[1] // bin_up_factor))

    for y in range(binned_mask_2d.shape[0]):
        for x in range(binned_mask_2d.shape[1]):
            value = True
            for y1 in range(bin_up_factor):
                for x1 in range(bin_up_factor):
                    padded_y = y*bin_up_factor + y1
                    padded_x = x*bin_up_factor + x1
                    if padded_mask_2d[padded_y, padded_x] == False:
                        value = False

            binned_mask_2d[y,x] = value

    return binned_mask_2d

@decorator_util.jit()
def padded_mask_2d_to_mask_1d_index_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Create a 2D array which maps every False entry of a 2D mask to its 1D mask array index 2D binned mask. Every \
    True entry is given a value -1.

    This uses the function *padded_mask_2d_to_mask_1d_index*, see this method for a more detailed description of the \
    mapping.

    This function first pads the mask using the same padding when computed a binned up mask.

    Parameters
    ----------
    mask_2d : ndarray
        The 2D mask that the mapping array is created for.

    Returns
    -------
    ndarray
        The 2D array mapping padded 2D mask entries to their 1D masked array indexes.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    mask_2d_to_mask_1d_index = padded_mask_2d_to_mask_1d_index_from_mask_2d(mask_2d=mask_2d)
    """

    padded_mask_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=mask_2d, bin_up_factor=bin_up_factor, pad_value=True)

    return mask_util.mask_2d_to_mask_1d_index_from_mask_2d(mask_2d=padded_mask_2d)

@decorator_util.jit()
def padded_mask_2d_to_binned_mask_1d_index_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Create a 2D array which maps every False entry of a 2D mask to its 1D binned mask index (created using the \
    *binned_up_mask_2d_from_mask_2d_and_bin_up_factor* method).

    We create an array the same shape as the 2D mask (after padding for the binnning up procedure), where each entry \
    gives the binned up mask's 1D masked array index.
    
    This is used as a convenience tool for creating arrays mapping between different grids and arrays.
    
    For example, if we had a 4x4 mask:
    
    [[False, False, False, False],
     [False, False, False, False],
     [ True,  True, False, False],
     [ True,  True, False, False]]
     
    For a bin_up_factor of 2, the resulting binned up mask is as follows (noting there is no padding in this example):
    
    [[False, False],
      [True, False]

    The mask_2d_to_binned_mask_1d_index is therefore:

    [[ 0,  0, 1, 1],
     [ 0,  0, 1, 1],
     [-1, -1, 2, 2],
     [-1, -1, 2, 2]]

    Parameters
    ----------
    mask_2d : ndarray
        The 2D mask that the binned mask 1d indexes are computing using
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The 2D array mapping 2D mask entries to their 1D binned masked array indexes.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    mask_to_binned_mask_2d =
        mask_2d_to_binned_mask_1d_index_from_mask_2d_and_bin_up_factor(mask_2d=mask_2d, bin_up_factor=3)
    """

    padded_mask_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=mask_2d, bin_up_factor=bin_up_factor, pad_value=True)

    binned_mask_2d = binned_up_mask_2d_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    mask_to_binned_mask_2d = np.full(
        fill_value=-1, shape=padded_mask_2d.shape)

    binned_mask_index = 0

    for bin_y in range(binned_mask_2d.shape[0]):
        for bin_x in range(binned_mask_2d.shape[1]):
            if binned_mask_2d[bin_y, bin_x] == False:
                for bin_y1 in range(bin_up_factor):
                    for bin_x1 in range(bin_up_factor):
                        mask_y = bin_y*bin_up_factor + bin_y1
                        mask_x = bin_x*bin_up_factor + bin_x1
                        if padded_mask_2d[mask_y, mask_x] == False:
                            mask_to_binned_mask_2d[mask_y, mask_x] = binned_mask_index

                binned_mask_index += 1

    return mask_to_binned_mask_2d

@decorator_util.jit()
def masked_array_1d_to_binned_masked_array_1d_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Create a 1D array which maps every (padded) masked index to its corresponding 1D index in the binned 1D \
    mask.

    This uses the convenience tools *padded_mask_2d_to_mask_1d* and *padded_mask_2d_to_binned_mask_1d* to \
    make the calculation simpler.

    For example, if we had a 4x4 mask:

    [[False, False, False, False],
     [False, False, False, False],
     [ True,  True, False, False],
     [ True,  True, False, False]]

    For a bin_up_factor of 2, the resulting binned up mask is as follows (noting there is no padding in this example):

    [[False, False],
      [True, False]

    The mask_2d_to_mask_1d_index is therefore:

    [[ 0,  1, 2,  3],
     [ 4,  5, 6,  7],
     [-1, -1, 8,  9],
     [-1, -1, 10, 11]]

    And the mask_2d_to_binned_mask_1d_index is therefore:

    [[ 0,  0, 1, 1],
     [ 0,  0, 1, 1],
     [-1, -1, 2, 2],
     [-1, -1, 2, 2]]

    Therefore, the binned_masked_array_1d_to_masked_array_1d would be:

    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2]

        This tells us that:
     - The first mask pixel maps to the first binned masked pixel (e.g. the 1D index of mask_2d after binning up).
     - The second mask pixel maps to the first binned masked pixel (e.g. the 1D index of mask_2d after binning up)
     - The third mask pixel maps to the second masked pixel (e.g. the 1D index of mask_2d after binning up)

    Parameters
    ----------
    mask_2d : ndarray
        The 2D mask that the binned mask 1d index mappings are computed using
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The 1D array mapping 1D binned mask entries to their corresponding 1D masked array index.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    mask_to_binned_mask_2d =
        binned_masked_array_1d_to_masked_array_1d_from_mask_2d_and_bin_up_factor(mask_2d=mask_2d, bin_up_factor=3)
    """

    padded_mask_2d = padded_binning_array_2d_from_array_2d_and_bin_up_factor(
        array_2d=mask_2d, bin_up_factor=bin_up_factor, pad_value=True)

    total_masked_pixels = mask_util.total_regular_pixels_from_mask(
        mask=padded_mask_2d)

    masked_array_1d_to_binned_masked_array_1d = np.zeros(
        shape=total_masked_pixels)

    padded_mask_2d_to_mask_1d_index = padded_mask_2d_to_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    padded_mask_2d_to_binned_mask_1d_index = padded_mask_2d_to_binned_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    for mask_y in range(padded_mask_2d_to_mask_1d_index.shape[0]):
        for mask_x in range(padded_mask_2d_to_mask_1d_index.shape[1]):
            if padded_mask_2d_to_mask_1d_index[mask_y, mask_x] >= 0:
                padded_mask_index = padded_mask_2d_to_mask_1d_index[mask_y, mask_x]
                binned_mask_index =  padded_mask_2d_to_binned_mask_1d_index[mask_y, mask_x]
                masked_array_1d_to_binned_masked_array_1d[padded_mask_index] = binned_mask_index

    return masked_array_1d_to_binned_masked_array_1d

@decorator_util.jit()
def binned_masked_array_1d_to_masked_array_1d_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Create a 1D array which maps every (padded) binned masked index to its correspond 1D index in the original 2D \
    mask that was binned up.

    Array indexing starts from the top-left and goes rightwards and downwards. The top-left pixel of each mask is \
    used before binning up.

    This uses the convenience tools *padded_mask_2d_to_mask_1d* and *padded_mask_2d_to_binned_mask_1d* to \
    make the calculation simpler.

    For example, if we had a 4x4 mask:

    [[False, False, False, False],
     [False, False, False, False],
     [ True,  True, False, False],
     [ True,  True, False, False]]

    For a bin_up_factor of 2, the resulting binned up mask is as follows (noting there is no padding in this example):

    [[False, False],
      [True, False]

    The mask_2d_to_mask_1d_index is therefore:

    [[ 0,  1, 2,  3],
     [ 4,  5, 6,  7],
     [-1, -1, 8,  9],
     [-1, -1, 10, 11]]

    And the mask_2d_to_binned_mask_1d_index is therefore:

    [[ 0,  0, 1, 1],
     [ 0,  0, 1, 1],
     [-1, -1, 2, 2],
     [-1, -1, 2, 2]]

    Therefore, the binned_masked_array_1d_to_masked_array_1d would be:

    [0, 2, 8]

    This tells us that:
     - The first binned mask pixel maps to the first masked pixel (e.g. the 1D index of mask_2d).
     - The second binned mask pixel maps to the third masked pixel (e.g. the 1D index of mask_2d)
     - The third binned mask pixel maps to the ninth masked pixel (e.g. the 1D index of mask_2d)

    Parameters
    ----------
    mask_2d : ndarray
        The 2D mask that the binned mask 1d index mappings are computed using
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The 1D array mapping 1D binned mask entries to their corresponding 1D masked array index.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    mask_to_binned_mask_2d =
        binned_masked_array_1d_to_masked_array_1d_from_mask_2d_and_bin_up_factor(mask_2d=mask_2d, bin_up_factor=3)
    """

    binned_up_mask_2d = binned_up_mask_2d_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    total_binned_masked_pixels = mask_util.total_regular_pixels_from_mask(
        mask=binned_up_mask_2d)

    binned_masked_array_1d_to_masked_array_1d = -1*np.ones(total_binned_masked_pixels)

    padded_mask_2d_to_mask_1d_index = padded_mask_2d_to_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    padded_mask_2d_to_binned_mask_1d_index = padded_mask_2d_to_binned_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    for mask_y in range(padded_mask_2d_to_mask_1d_index.shape[0]):
        for mask_x in range(padded_mask_2d_to_mask_1d_index.shape[1]):
            if padded_mask_2d_to_mask_1d_index[mask_y, mask_x] >= 0:
                binned_mask_index =  padded_mask_2d_to_binned_mask_1d_index[mask_y, mask_x]
                if binned_masked_array_1d_to_masked_array_1d[binned_mask_index] == -1:
                    padded_mask_index = padded_mask_2d_to_mask_1d_index[mask_y, mask_x]
                    binned_masked_array_1d_to_masked_array_1d[binned_mask_index] = \
                        padded_mask_index

    return binned_masked_array_1d_to_masked_array_1d

# @decorator_util.jit()
def binned_masked_array_1d_to_masked_array_1d_all_from_mask_2d_and_bin_up_factor(mask_2d, bin_up_factor):
    """Create a 2D array which maps every (padded) binned masked index to all of the corresponding 1D indexes of the \
    the original 2D mask that was binned up.

    Array indexing starts from the top-left and goes rightwards and downwards. The top-left pixel of each mask is \
    used before binning up. Minus one's are used for mapping which go to masked values with True.

    This uses the convenience tools *padded_mask_2d_to_mask_1d* and *padded_mask_2d_to_binned_mask_1d* to \
    make the calculation simpler.

    For example, if we had a 4x4 mask:

    [[False, False, False, False],
     [False, False, False, False],
     [ True,  True, False, False],
     [ True,  True, True, False]]

    For a bin_up_factor of 2, the resulting binned up mask is as follows (noting there is no padding in this example):

    [[False, False],
      [True, False]

    The mask_2d_to_mask_1d_index is therefore:

    [[ 0,  1, 2,  3],
     [ 4,  5, 6,  7],
     [-1, -1, 8,  9],
     [-1, -1, -1, 10]]

    And the mask_2d_to_binned_mask_1d_index is therefore:

    [[ 0,  0, 1, 1],
     [ 0,  0, 1, 1],
     [-1, -1, 2, 2],
     [-1, -1, 2, 2]]

    Therefore, the binned_masked_array_1d_to_masked_array_1d_all would be:

    [[0, 1, 4, 5],
     [2, 3, 6, 7]]
     [8, 9, 10, -1]]

    This tells us that:
     - The first binned mask pixel maps to the first, second, fifth and sixth masked pixels.
     - The second binned mask pixel maps to the third, fourth, seventh and eighth masked pixels
     - The third binned mask pixel maps to the ninth, tenth and eleventh masked pixels (The fourth masked pixel it \
       maps to is a *True* value and therefore masked.)

    Parameters
    ----------
    mask_2d : ndarray
        The 2D mask that the binned mask 1d index mappings are computed using
    bin_up_factor : int
        The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).

    Returns
    -------
    ndarray
        The 1D array mapping 1D binned mask entries to their corresponding 1D masked array index.

    Examples
    --------
    mask_2d = np.full(fill_value=False, shape=(9,9))
    mask_to_binned_mask_2d =
        binned_masked_array_1d_to_masked_array_1d_from_mask_2d_and_bin_up_factor(mask_2d=mask_2d, bin_up_factor=3)
    """

    binned_up_mask_2d = binned_up_mask_2d_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    total_binned_masked_pixels = mask_util.total_regular_pixels_from_mask(
        mask=binned_up_mask_2d)

    binned_masked_array_1d_to_masked_array_1d_all = -1*np.ones((total_binned_masked_pixels, bin_up_factor**2))

    binned_masked_array_1d_sizes = np.zeros(total_binned_masked_pixels)

    padded_mask_2d_to_mask_1d_index = padded_mask_2d_to_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    padded_mask_2d_to_binned_mask_1d_index = padded_mask_2d_to_binned_mask_1d_index_from_mask_2d_and_bin_up_factor(
        mask_2d=mask_2d, bin_up_factor=bin_up_factor)

    for mask_y in range(padded_mask_2d_to_mask_1d_index.shape[0]):
        for mask_x in range(padded_mask_2d_to_mask_1d_index.shape[1]):
            if padded_mask_2d_to_mask_1d_index[mask_y, mask_x] >= 0:

                binned_mask_index = padded_mask_2d_to_binned_mask_1d_index[mask_y, mask_x]
                binned_mask_count = int(binned_masked_array_1d_sizes[binned_mask_index])
                padded_mask_index = padded_mask_2d_to_mask_1d_index[mask_y, mask_x]

                binned_masked_array_1d_to_masked_array_1d_all[binned_mask_index, binned_mask_count] = \
                    padded_mask_index

                binned_masked_array_1d_sizes[binned_mask_index] += 1

    return binned_masked_array_1d_to_masked_array_1d_all, binned_masked_array_1d_sizes