import numpy as np
import image

def setup_data(mask, data):
    """ Given an image.Mask, convert a 2d data of data values to a 1D vector, structured for efficient lens modeling \
    calculations.

    Parameters
    ----------
    data : ndarray
        A 2D data of data, e.g. the image, noise-map, etc.
    mask : image.Mask
        The image mask containing the pixels we are computing the coordinates of and the image dimensions / pixel scale.

    Returns
    -------
    One-dimensional data containing data of data.
    """
    image_pixels = mask.pixels_in_mask
    data_1d = np.zeros(shape=(image_pixels))
    data_count = 0
    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                data_1d[data_count] = data[y, x]
                data_count += 1

    return data_1d

def setup_mapper_2d(mask):
    """ Given an image.Mask, setup an array that can be used to map the input masks coordinates back to their 2D image \
    pixels.

    Parameters
    ----------
    mask : image.Mask
        The image mask containing the pixels we are computing the coordinates of and the image dimensions / pixel scale.
    """
    image_pixels = mask.pixels_in_mask
    mapper_2d = np.zeros(shape=(image_pixels, 2), dtype=int)
    data_count = 0
    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                mapper_2d[data_count, :] = np.array([y, x])
                data_count += 1

    return mapper_2d

def setup_image_coordinates(mask):
    """ Given an image.Mask, compute the arc second image_coordinates at the center of every unmasked pixel.

    This is defined from the top-left corner, such that pixels in the top-left corner of the image have a negative \
    x value for and positive y value in arc seconds.

    The results are returned as a 1D array, structured for efficient lens modeling calculations.

    Parameters
    ----------
    mask : image.Mask
        The image mask containing the pixels we are computing the image_coordinates of and the image dimensions / pixel scale.

    Returns
    -------
    One-dimensional array containing the image image_coordinates of each image pixel in the mask.
    """
    coordinates = image.arc_seconds_coordinates_of_array(mask.pixel_dimensions, mask.pixel_scale)

    pixels = mask.pixels_in_mask
    image_coordinates = np.zeros(shape=(pixels, 2))
    pixel_count = 0

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                image_coordinates[pixel_count, :] = coordinates[y,x]
                pixel_count += 1

    return image_coordinates

def x_sub_pixel_to_coordinate(x_sub_pixel, x_coordinate, pixel_scale, sub_grid_size):
    """Convert a coordinate on the regular image-pixel grid to a sub-coordinate, using the pixel scale and sub-grid \
    size"""

    half = pixel_scale / 2
    step = pixel_scale / (sub_grid_size + 1)

    return x_coordinate - half + (x_sub_pixel + 1) * step

def y_sub_pixel_to_coordinate(y_sub_pixel, y_coordinate, pixel_scale, sub_grid_size):
    """Convert a coordinate on the regular image-pixel grid to a sub-coordinate, using the pixel scale and sub-grid \
    size"""

    half = pixel_scale / 2
    step = pixel_scale / (sub_grid_size + 1)

    return y_coordinate + half - (y_sub_pixel + 1) * step

def setup_sub_coordinates(mask, sub_grid_size):
    """
    Given an image.Mask, compute the arc second coordinates of every unmasked sub image-pixel.

    This is defined from the top-left corner, such that pixels in the top-left corner of the image have a negative x
    value for and positive y value in arc seconds. Sub-pixel are defined from the top-left hand corner of each \
    image pixel.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the blurring region around and the image dimensions / pixel scale.
    sub_grid_size : int
        The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.

    Returns
    -------
    One-dimensional array containing the sub-image coordinates of each image pixel in the mask.
    """

    cen = image.central_pixel(mask.pixel_dimensions)

    image_pixels = mask.pixels_in_mask
    image_pixel_count = 0

    image_sub_grid_coordinates = np.zeros(shape=(image_pixels, sub_grid_size**2, 2))

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                x_coordinate = image.x_pixel_to_arc_seconds(x, cen[1], mask.pixel_scale)
                y_coordinate = image.y_pixel_to_arc_seconds(y, cen[0], mask.pixel_scale)
                sub_pixel_count = 0

                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):

                        image_sub_grid_coordinates[image_pixel_count, sub_pixel_count, 0] = \
                            x_sub_pixel_to_coordinate(x1, x_coordinate, mask.pixel_scale, sub_grid_size)

                        image_sub_grid_coordinates[image_pixel_count, sub_pixel_count, 1] = \
                            y_sub_pixel_to_coordinate(y1, y_coordinate, mask.pixel_scale, sub_grid_size)

                        sub_pixel_count += 1

                image_pixel_count += 1

    return image_sub_grid_coordinates

def setup_blurring_coordinates(mask, psf_size):
    """Given an image.Mask, compute its blurring mask and use this to find the coordinates of all regions in the \
    bluring mask.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the blurring region around and the image dimensions / pixel scale.
    psf_size : (int, int)
        The size of the kernel which defines the blurring region (e.g. the pixel dimensions of the PSF kernel)


    """
    blurring_mask = setup_blurring_mask(mask, psf_size)
    return setup_image_coordinates(blurring_mask)

def setup_blurring_mask(mask, psf_size):
    """Given an image.Mask, compute its blurring mask, defined as all pixels which are outside of the image mask but \
    will have their light blurred into the mask during PSF convolution. Thus, their light must be computed during \
    the analysis to ensure accurate PSF blurring.

    The blurring_region_size is a tuple describing the rectangular pixel dimensions of the blurring kernel. \
    Therefore, it maps directly to the size of the PSF kernel of an image.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the blurring region around and the image dimensions / pixel scale.
    psf_size : (int, int)
        The size of the kernel which defines the blurring region (e.g. the pixel dimensions of the PSF kernel)

    Returns
    -------
    blurring_mask : image.Mask
        A mask where every False value is a pixel which is within the mask's blurring region.

     """

    if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
        raise image.MaskException("psf_size of exterior region must be odd")

    blurring_mask = np.ones(mask.pixel_dimensions)

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                for y1 in range((-psf_size[1] + 1) // 2, (psf_size[1] + 1) // 2):
                    for x1 in range((-psf_size[0] + 1) // 2, (psf_size[0] + 1) // 2):
                        if 0 <= y + y1 <= mask.pixel_dimensions[0] - 1 \
                                and 0 <= x + x1 <= mask.pixel_dimensions[1] - 1:
                            if mask[y + y1, x + x1] == True:
                                blurring_mask[y + y1, x + x1] = False
                        else:
                            raise image.MaskException("setup_blurring_mask extends beynod the size of the mask - pad the image"
                                "before masking")

    return image.Mask.from_array(blurring_mask, mask.pixel_scale)

def setup_border_pixels(mask):
    """Given an image.Mask, compute its border image pixels, defined as all pixels which are on the edge of the mask \
    and therefore neighboring a pixel with a *True* value.

    The border pixels are used to relocate highly demagnified traced image pixels in the source-plane to its edge.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the border pixels of and the image dimensions / pixel scale.

    Returns
    -------
    border_pixels : ndarray
        The border image pixels, where each entry gives the 1D index of the image pixel in the mask.
    """

    # TODO : This border only works for circular / elliptical masks which do not have masked image pixels in their
    # TODO : center (like an annulus). Do we need a separate routine for annuli masks?
    border_pixels = np.empty(0)
    image_pixel_index = 0

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                if mask[y+1,x] == 1 or mask[y-1,x] == 1 or mask[y,x+1] == 1 or mask[y,x-1] == 1 or \
                        mask[y+1,x+1] == 1 or mask[y+1, x-1] == 1 or mask[y-1, x+1] == 1 or mask[y-1,x-1] == 1:
                    border_pixels = np.append(border_pixels, image_pixel_index)
                image_pixel_index += 1

    return border_pixels

def setup_sparse_pixels(mask, sparse_grid_size):
    """Given an image.Mask, compute the sparse cluster image pixels, defined as the sub-set of image-pixels used \
    to perform KMeans clustering (this is used purely for speeding up the KMeans clustering algorithim).

    This sparse grid is a uniform subsample of the masked image and is computed by only including image pixels \
    which, when divided by the sparse_grid_size, do not give a remainder.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of and the image dimensions / pixel scale.
    sparse_grid_size : int
        The spacing of the sparse image pixel grid (e.g. a value of 2 will compute a sparse grid of pixels which \
        are two pixels apart)

    Returns
    -------
    sparse_to_image : ndarray
        The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
        of the image pixel in the mask.
    image_to_sparse : ndarray
        The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D \
        index of the sparse pixel in sparse_pixel arrays.
    """

    sparse_mask = setup_uniform_sparse_mask(mask, sparse_grid_size)
    sparse_index_image = setup_sparse_index_image(mask, sparse_mask)
    sparse_to_image = setup_sparse_to_image(mask, sparse_mask)
    image_to_sparse = setup_image_to_sparse(mask, sparse_mask, sparse_index_image)

    return sparse_to_image, image_to_sparse

def setup_uniform_sparse_mask(mask, sparse_grid_size):
    """Setup a two-dimensional sparse mask of image pixels, by keeping all image pixels which do not give a remainder \
    when divided by the sub-grid size. """
    sparse_mask = np.ones(mask.pixel_dimensions)

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                    sparse_mask[y, x] = False

    return image.Mask.from_array(sparse_mask, mask.pixel_scale)

def setup_sparse_index_image(mask, sparse_mask):
    """Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.

     This is used for computing the image_to_sparse vector, whereby each image pixel is paired to the sparse pixel \
     in this image via a neighbor search."""

    sparse_index_2d = np.zeros(mask.pixel_dimensions)
    sparse_pixel_index = 0

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if sparse_mask[y,x] == False:
                sparse_pixel_index += 1
                sparse_index_2d[y,x] = sparse_pixel_index

    return sparse_index_2d

def setup_sparse_to_image(mask, sparse_mask):
    """Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
    pixels.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of and the image dimensions / pixel scale.
    sparse_mask : ndarray
        The two-dimensional boolean image of the sparse grid.

    Returns
    -------
    sparse_to_image : ndarray
        The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
        of the image pixel in the mask.
    """
    sparse_to_image = np.empty(0)
    image_pixel_index = 0

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):

            if sparse_mask[y, x] == False:
                sparse_to_image = np.append(sparse_to_image, image_pixel_index)

            if mask[y, x] == False:
                image_pixel_index += 1

    return sparse_to_image

def setup_image_to_sparse(mask, sparse_mask, sparse_index_image):
    """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.

    This is performed by going to each image pixel in the *mask*, and pairing it with its nearest neighboring pixel \
    in the *sparse_mask*. The index of the *sparse_mask* pixel is drawn from the *sparse_index_image*. This \
    neighbor search continue grows larger and larger around a pixel, until a pixel contained in the *sparse_mask* is \
    successfully found.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of and the image dimensions / pixel scale.
    sparse_mask : ndarray
        The two-dimensional boolean image of the sparse grid.

    Returns
    -------
    image_to_sparse : ndarray
        The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D \
        index of the sparse pixel in sparse_pixel arrays.

    """
    image_to_sparse = np.empty(0)

    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                iboarder = 0
                pixel_match = False
                while pixel_match == False:
                    for y1 in range(y-iboarder, y+iboarder+1):
                        for x1 in range(x-iboarder, x+iboarder+1):
                            if y1 >= 0 and y1 < mask.pixel_dimensions[0] and x1 >= 0 and x1 < mask.pixel_dimensions[1]:
                                if sparse_mask[y1, x1] == False and pixel_match == False:
                                    image_to_sparse = np.append(image_to_sparse, sparse_index_image[y1,x1]-1)
                                    pixel_match = True

                    iboarder += 1
                    if iboarder == 100:
                        raise image.MaskException('setup_image_to_sparse - Stuck in infinite loop')

    return image_to_sparse

class AnalysisData(object):

    def __init__(self, mask, image, noise, psf, sub_grid_size=2):
        """The core grouping of lens modeling data, including the image data, noise-map and psf. Optional \
        inputs (e.g. effective exposure time map / positional image pixels) have their functionality automatically \
        switched on or off depending on if they are included.

        A mask must be supplied, which converts all 2D image quantities to data vectors. These vectors are designed to \
        provide optimal lens modeling efficiency. Image region vectors are also set-up, which describe specific \
        regions of the image. These are used for specific calculations, like the image sub-grid, and to provide \

        Parameters
        ----------
        mask : image.Mask
            The image mask, where False indicates a pixel is included in the analysis.
        image : image.Image
            The image data, in electrons per second.
        noise : image.Noise
            The image noise-map, in variances in electrons per second.
        psf : image.PSF
            The image PSF
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        self.image = AnalysisImage(mask, image)
        self.noise = setup_data(mask, noise)
        self.psf = psf
        self.coordinates = setup_image_coordinates(mask)
        self.sub_coordinates = setup_sub_coordinates(mask, sub_grid_size)
        self.blurring_coordinates = setup_blurring_coordinates(mask, self.psf.shape)
        self.border_pixels = setup_border_pixels(mask)


class AnalysisArray(np.ndarray):

    def __new__(cls, mask, data):
        """

        Parameters
        ----------
        mask_array : ndarray
            The boolean array of masked pixels (False = pixel is not masked and included in analysis)

        Returns
        -------
            An empty array
        """
        data = setup_data(mask, data).view(cls)
        data.mapper_2d = setup_mapper_2d(mask)
        data.shape_2d = mask.shape
        return data

    def map_to_2d(self):
        array = np.zeros((self.shape_2d))
        for data_count, [y, x] in enumerate(self.mapper_2d):
            array[y,x] = self[data_count]

        return array


# TODO : This may be where we put our hyper-parameter functions for the image and noise maps. E.g. for noise map \
# TODO : scaling, we could have an AnalysisNoise class with functions def scale_lens_noise, etc. I'll decide on this \
# TODO : As we continue to develop the analysis code.

# TODO : Can we handle these classes using inheritance of Analysis Array? I can't figue out how but this works...

class AnalysisImage(np.ndarray):

    def __new__(cls, mask, image):
        return AnalysisArray(mask, image)