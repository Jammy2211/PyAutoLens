import numpy as np
import image

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

def setup_image_sub_grid_coordinates(mask, pixel_scale, sub_grid_size):
    """
    Given a mask, the dimensions of the observed image and its pixel_scale, compute the arc second coordinates of every \
    sub image-pixel. This is defined from the top-left corner, such that pixels in the top-left corner of the image
    have a negative x value for and positive y value in arc seconds. Sub-pixel are defined from the top-left hand \
    corner of each image pixel.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the blurring region around.
    pixel_dimensions : (int, int)
        The dimensions of the input array
    pixel_scale : float
        The arcsecond to pixel conversion factor of the array.
    sub_grid_size : int
        The sub_grid_size x sub_grid_size of the sub-grid of each image pixel.

    Returns
    -------
    One-dimensional array containing the sub-image coordinates of each image pixel in the mask.
    """

    image_pixels = np.sum(mask)
    pixel_dimensions = mask.shape

    cen = image.central_pixel(pixel_dimensions)

    image_sub_grid_coordinates = np.zeros(shape=(image_pixels, sub_grid_size**2, 2))

    image_pixel_count = 0

    for y in range(pixel_dimensions[0]):
        for x in range(pixel_dimensions[1]):
            if mask is None or mask[y, x] == True:
                x_coordinate = image.x_pixel_to_arc_seconds(x, cen[1], pixel_scale)
                y_coordinate = image.y_pixel_to_arc_seconds(y, cen[0], pixel_scale)
                sub_pixel_count = 0

                for y1 in range(sub_grid_size):
                    for x1 in range(sub_grid_size):

                        image_sub_grid_coordinates[image_pixel_count, sub_pixel_count, 0] = \
                            x_sub_pixel_to_coordinate(x1, x_coordinate, pixel_scale, sub_grid_size)

                        image_sub_grid_coordinates[image_pixel_count, sub_pixel_count, 1] = \
                            y_sub_pixel_to_coordinate(y1, y_coordinate, pixel_scale, sub_grid_size)

                        sub_pixel_count += 1

                image_pixel_count += 1

    return image_sub_grid_coordinates

def setup_blurring_region(mask, blurring_region_size):
    """Compute the blurring region of a mask, where the blurring region is defined as all pixels which are outside \
     of the mask but will have their light blurred into the mask during PSF convolution. Thus, their light must be \
     computed during the analysis to ensure accurate PSF blurring.

     The blurring_region_size is a tuple describing the rectangular pixel dimensions of the blurring kernel. \
     Therefore, it maps directly to the size of the PSF kernel of an image.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the blurring region around.
    blurring_region_size : int
        The size of the kernel which defines the blurring region (e.g. the pixel dimensions of the PSF kernel)

    Returns
    -------
    setup_blurring_region : numpy.ma
        A mask where every True value is a pixel which is within the mask's blurring region.

     """

    if blurring_region_size[0] % 2 == 0 or blurring_region_size[1] % 2 == 0:
        raise MaskException("blurring_region_size of exterior region must be odd")

    image_dimensions_pixels = mask.shape
    blurring_region = np.zeros(image_dimensions_pixels)

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):
            if mask[y, x]:
                for y1 in range((-blurring_region_size[1] + 1) // 2, (blurring_region_size[1] + 1) // 2):
                    for x1 in range((-blurring_region_size[0] + 1) // 2, (blurring_region_size[0] + 1) // 2):
                        if 0 <= y + y1 <= image_dimensions_pixels[0] - 1 \
                                and 0 <= x + x1 <= image_dimensions_pixels[1] - 1:
                            if not mask[y + y1, x + x1]:
                                blurring_region[y + y1, x + x1] = True
                        else:
                            raise MaskException(
                                "setup_blurring_region extends beynod the size of the mask - pad the image"
                                "before masking")

    return blurring_region

def setup_border_pixels(mask):
    """Compute the border image pixels of a mask, where the border pixels are defined as all pixels which are on the
     edge of the mask and neighboring a pixel with a  *False* value.

     The border pixels are used to relocate highly demagnified traced image pixels in the source-plane to its edge.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the border pixels of.

    Returns
    -------
    border_pixels : ndarray
        The border image pixels, where each entry gives the 1D index of the image pixel in the mask.
    """

    # TODO : This border only works for circular / elliptical masks which do not have masked image pixels in their
    # TODO : center (like an annulus). Do we need a separate routine for annuli masks?

    image_dimensions_pixels = mask.shape
    border_pixels = np.empty(0)
    image_pixel_index = 0

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):
            if mask[y, x]:
                if mask[y+1,x] == 0 or mask[y-1,x] == 0 or mask[y,x+1] == 0 or mask[y,x-1] == 0 or \
                        mask[y+1,x+1] == 0 or mask[y+1, x-1] == 0 or mask[y-1, x+1] == 0 or mask[y-1,x-1] == 0:
                    border_pixels = np.append(border_pixels, image_pixel_index)
                image_pixel_index += 1

    return border_pixels

def setup_sparse_pixels(mask, sparse_grid_size):
    """Compute the sparse cluster image pixels in a mask, where the sparse cluster image pixels are the sub-set of \
    image-pixels used within the mask to perform KMeans clustering (this is used purely for speeding up the \
    KMeans clustering algorithim).

    This sparse grid is a uniform subsample of the masked image and is computed by only including image pixels \
    which, when divided by the sparse_grid_size, do not give a remainder.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of.
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

    sparse_mask = setup_sparse_mask(mask, sparse_grid_size)
    sparse_to_image = setup_sparse_to_image(mask, sparse_mask)
    image_to_sparse = setup_image_to_sparse(mask, sparse_mask)

    return sparse_to_image, image_to_sparse

def setup_sparse_mask(mask, sparse_grid_size):
    """Setup a two-dimensional sparse mask of image pixels, by keeping all image pixels which do not give a remainder \
    when divided by the sub-grid size. """

    image_dimensions_pixels = mask.shape

    sparse_mask = np.zeros(image_dimensions_pixels)

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):
            if mask[y, x]:
                if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                    sparse_mask[y, x] = 1

    return np.ma.make_mask(sparse_mask)

def setup_sparse_to_image(mask, sparse_mask):
    """Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
    pixels.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of.
    sparse_mask : ndarray
        The two-dimensional boolean image of the sparse grid.

    Returns
    -------
    sparse_to_image : ndarray
        The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
        of the image pixel in the mask.
    """
    image_dimensions_pixels = mask.shape

    sparse_to_image = np.empty(0)
    image_pixel_index = 0

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):

            if sparse_mask[y, x]:
                sparse_to_image = np.append(sparse_to_image, image_pixel_index)

            if mask[y, x]:
                image_pixel_index += 1

    return sparse_to_image

def setup_image_to_sparse(mask, sparse_mask):
    """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.

    Parameters
    ----------
    mask : image.Mask
        The image mask we are finding the sparse clustering pixels of.
    sparse_mask : ndarray
        The two-dimensional boolean image of the sparse grid.

    Returns
    -------
    image_to_sparse : ndarray
        The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D \
        index of the sparse pixel in sparse_pixel arrays.

    """

    image_dimensions_pixels = mask.shape
    sparse_index_2d = np.zeros(image_dimensions_pixels)
    sparse_pixel_index = 0

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):
            if sparse_mask[y,x]:
                sparse_pixel_index += 1
                sparse_index_2d[y,x] = sparse_pixel_index

    image_to_sparse = np.empty(0)

    for y in range(image_dimensions_pixels[0]):
        for x in range(image_dimensions_pixels[1]):
            if mask[y, x]:
                iboarder = 0
                pixel_match = False
                while pixel_match == False:
                    for y1 in range(y-iboarder, y+iboarder+1):
                        for x1 in range(x-iboarder, x+iboarder+1):
                            if y1 >= 0 and y1 < image_dimensions_pixels[0] and x1 >= 0 and x1 < image_dimensions_pixels[1]:
                                if sparse_mask[y1, x1] and pixel_match == False:
                                    image_to_sparse = np.append(image_to_sparse, sparse_index_2d[y1,x1]-1)
                                    pixel_match = True

                    iboarder += 1
                    if iboarder == 100:
                        raise MaskException('setup_image_to_sparse - Stuck in infinite loop')

    return image_to_sparse


class Mask(object):
    """Abstract Class for preparing and storing the image mask used for the AutoLens analysis"""

    @classmethod
    def mask(cls, arc_second_dimensions, pixel_scale, function, centre):
        """

        Parameters
        ----------
        centre: (float, float)
            The centre in arc seconds
        function: function(x, y) -> Bool
            A function that determines what the value of a mask should be at particular coordinates
        pixel_scale: float
            The size of a pixel in arc seconds
        arc_second_dimensions: (float, float)
            The spatial dimensions of the mask in arc seconds

        Returns
        -------
            An empty array
        """
        pixel_dimensions = image.arc_second_dimensions_to_pixel(arc_second_dimensions, pixel_scale)
        array = np.zeros((int(pixel_dimensions[0]), int(pixel_dimensions[1])))

        central_pixel_coords = image.central_pixel(pixel_dimensions)
        for i in range(int(pixel_dimensions[0])):
            for j in range(int(pixel_dimensions[1])):
                # Convert from pixel coordinates to image coordinates
                x_pix = pixel_scale * (i - central_pixel_coords[0]) - centre[0]
                y_pix = pixel_scale * (j - central_pixel_coords[1]) - centre[1]

                array[i, j] = function(x_pix, y_pix)

        return np.ma.make_mask(array)

    @classmethod
    def circular(cls, arc_second_dimensions, pixel_scale, radius, centre=(0., 0.)):
        """

        Parameters
        ----------
        centre: (float, float)
            The centre in image coordinates in arc seconds
        arc_second_dimensions : (int, int)
            The dimensions of the image (x, y) in arc seconds
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        radius : float
            The radius of the circle (arc seconds)
        """

        def is_within_radius(x_pix, y_pix):
            radius_arc = np.sqrt(x_pix ** 2 + y_pix ** 2)
            return radius_arc <= radius

        return Mask.mask(arc_second_dimensions, pixel_scale, is_within_radius, centre)

    @classmethod
    def annular(cls, arc_second_dimensions, pixel_scale, inner_radius, outer_radius, centre=(0., 0.)):
        """

        Parameters
        ----------
        centre: (float, float)
            The centre in arc seconds
        arc_second_dimensions : (int, int)
            The dimensions of the image in arcs seconds
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        inner_radius : float
            The inner radius of the circular annulus (arc seconds
        outer_radius : float
            The outer radius of the circular annulus (arc seconds)
        """

        def is_within_radii(x_pix, y_pix):
            radius_arc = np.sqrt(x_pix ** 2 + y_pix ** 2)
            return outer_radius >= radius_arc >= inner_radius

        return Mask.mask(arc_second_dimensions, pixel_scale, is_within_radii, centre)

class MaskException(Exception):
    pass
