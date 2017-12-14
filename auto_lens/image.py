from scipy.stats import norm
from astropy.io import fits
import os
import logging
from functools import wraps
import numpy as np

# TODO: this gives us a logger that will print stuff with the name of the module
logging.basicConfig()
logger = logging.getLogger(__name__)

data_path = "{}/../../data/prep_lens/".format(os.path.dirname(os.path.realpath(__file__)))


def numpy_array_from_fits(file_path, hdu):
    hdu_list = fits.open(file_path)  # Open the fits file
    return np.array(hdu_list[hdu].data)


def pixel_dimensions_to_arc_seconds(pixel_dimensions, pixel_scale):
    return tuple(map(lambda d: d * pixel_scale, pixel_dimensions))


def arc_second_dimensions_to_pixel(arc_second_dimensions, pixel_scale):
    return tuple(map(lambda d: d / pixel_scale, arc_second_dimensions))


def central_pixel(pixel_dimensions):
    return tuple(map(lambda d: (float(d + 1) / 2) - 1, pixel_dimensions))


def copy_attributes(old_obj, new_obj):
    if hasattr(old_obj, "__dict__"):
        for t in old_obj.__dict__.items():
            setattr(new_obj, t[0], t[1])
    return new_obj


def keep_attributes(func):
    """
    
    Parameters
    ----------
    func: function(T:ndarray) -> T
        A function that takes a child of ndarray and returns an instance of that class
    Returns
    -------
    func: function(T:ndarray) -> T
        A function that takes a child of ndarray and returns the same class with associated instance attributes
    """

    @wraps(func)
    def wrapper(array, *args, **kwargs):
        """
        
        Parameters
        ----------
        array: T:ndarray
            A ndarray or child thereof
        args
        kwargs

        Returns
        -------
        array: T:ndarray
            A new instance of the same class that has been trimmed and retains all the instance attributes of the
            original array
        """
        new_array = func(array, *args, **kwargs).view(array.__class__)
        if hasattr(array, "__dict__"):
            copy_attributes(array, new_array)

        return new_array

    return wrapper


# TODO: It seemed to meet that many of these functions are best made general. They really can apply to any array.
@keep_attributes
def trim(array, pixel_dimensions):
    """ Trim the data array to a new size around its central pixel.
    NOTE: The centre of the array cannot be shifted. Therefore, even arrays are trimmed to even arrays
    (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).
    Parameters
    ----------
    array: ndarray (or Noise or PSF)
        The image array
    pixel_dimensions : (int, int)
        The new pixel dimensions of the trimmed data-array
    """
    shape = array.shape
    if pixel_dimensions[0] > shape[0]:
        raise ValueError('image.Data.trim_data - You have specified a new x_size bigger than the data array')
    elif pixel_dimensions[1] > shape[1]:
        raise ValueError('image.Data.trim_data - You have specified a new y_size bigger than the data array')
    x_trim = int((shape[0] - pixel_dimensions[0]) / 2)
    y_trim = int((shape[1] - pixel_dimensions[1]) / 2)
    array = array[x_trim:shape[0] - x_trim, y_trim:shape[1] - y_trim]
    if shape[0] != pixel_dimensions[0]:
        logger.debug(
            'image.data.trim_data - Your specified x_size was odd (even) when the image x dimension is even (odd)')
        logger.debug(
            'The method has automatically used x_size+1 to ensure the image is not miscentred by a half-pixel.')
    elif shape[1] != pixel_dimensions[1]:
        logger.debug(
            'image.data.trim_data - Your specified y_size was odd (even) when the image y dimension is even (odd)')
        logger.debug(
            'The method has automatically used y_size+1 to ensure the image is not miscentred by a half-pixel.')
    return array


@keep_attributes
def pad(array, pixel_dimensions):
    """ Pad the data array with zeros around its central pixel.
    NOTE: The centre of the array cannot be shifted. Therefore, even arrays are padded to even arrays
    (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).
    Parameters
    ----------
    array: ndarray (or Noise or PSF)
        The image array
    pixel_dimensions : (int, int)
        The new pixel dimension of the data-array
    """
    shape = array.shape
    if pixel_dimensions[0] < shape[0]:
        raise ValueError('image.Data.pad_data - You have specified a new x_size smaller than the data array')
    elif pixel_dimensions[1] < shape[1]:
        raise ValueError('image.Data.pad_data - You have specified a new y_size smaller than the data array')
    x_pad = int((pixel_dimensions[0] - shape[0] + 1) / 2)
    y_pad = int((pixel_dimensions[1] - shape[1] + 1) / 2)
    return np.pad(array, ((x_pad, y_pad), (x_pad, y_pad)), 'constant')


def output_for_fortran(array, image_name, path=data_path):
    """ Outputs the data-array for the Fortran AutoLens code. This will ultimately be removed so you can ignore
    and I've not bothered with unit-tests.
    Parameters
    ----------
    array: ndarray (or Noise or PSF)
        The image array
    path : str
        The directory the files are output too
    image_name : str
        The name of the image for this file
    """
    # TODO: Here a default option was necessary else the code would crash if a different type was passed in
    if isinstance(array, PSF):
        file_path = path + image_name + "PSF.dat"
    elif isinstance(array, Noise):
        file_path = path + image_name + "BaselineNoise.dat"
    else:
        file_path = path + image_name + ".dat"
    shape = array.shape

    # TODO: This convention is nice. file f exists in the scope but if closed after execution is finished
    with open(file_path, "w+") as f:
        for ix, x in enumerate(range(shape[0])):
            for iy, y in enumerate(range(shape[1])):
                line = str(round(float(ix + 1), 2))
                line += ' ' * (8 - len(line))
                line += str(round(float(iy + 1), 2))
                line += ' ' * (16 - len(line))
                line += str(float(array.data[ix][iy])) + '\n'
                f.write(line)


class Image(np.ndarray):
    # TODO: this is a bit of magic. __new__ gets called before __init__. In this case we can use it to initialise an
    # TODO: ndarray with some extra attributes
    def __new__(cls, array, pixel_scale, sky_background_level=None, sky_background_noise=None):
        """
        Creates a new image, accounting for the fact that Image is a ndarray
        Parameters
        ----------
        array: ndarray
            The array of data
        pixel_scale: float
            The scale of an image pixel
        sky_background_level
        sky_background_noise

        Returns
        -------
            A new Image object
        """
        obj = np.asarray(array).view(cls)
        obj.pixel_scale = pixel_scale

        obj.sky_background_level = sky_background_level
        obj.sky_background_noise = sky_background_noise

        return obj

    def __array_finalize__(self, obj):
        """
        Used to pass data around for some Numpy functions
        Parameters
        ----------
        obj: Image
            The original image

        Returns
        -------
            The new image
        """
        if obj is not None:
            copy_attributes(obj, self)

    @property
    def central_pixels(self):
        return central_pixel(self.shape)

    @property
    def shape_arc_seconds(self):
        return pixel_dimensions_to_arc_seconds(self.shape, self.pixel_scale)

    @property
    def x_cen_pixel(self):
        return self.central_pixels[0]

    @property
    def y_cen_pixel(self):
        return self.central_pixels[1]

    # TODO: please can we use filename? It's pretty standard. file_name is very rare.
    @classmethod
    def from_fits(cls, filename, hdu, pixel_scale, sky_background_level=None, sky_background_noise=None,
                  path=data_path):
        """Load the image from a fits file.

        Parameters
        ----------
        filename : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        sky_background_level : float
            An estimate of the level of background sky in the image (electrons per second).
        sky_background_noise : float
            An estimate of the noise level in the background sky (electrons per second).
        path : str
            The directory path to the fits file
        """
        array = numpy_array_from_fits(path + filename, hdu)
        return Image(array, pixel_scale, sky_background_level,
                     sky_background_noise)

    def set_sky_via_edges(self, no_edges):
        """Estimate the background sky level and noise by binning pixels located at the edge(s) of an image into a
        histogram and fitting a Gaussian profile to this histogram. The mean (mu) of this Gaussian gives the background
        sky level, whereas the FWHM (sigma) gives the noise estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the backgroundd sky properties

        """

        edges = []

        for edge_no in range(no_edges):
            top_edge = self[edge_no, edge_no:self.shape[1] - edge_no]
            bottom_edge = self[self.shape[0] - 1 - edge_no, edge_no:self.shape[1] - edge_no]
            left_edge = self[edge_no + 1:self.shape[0] - 1 - edge_no, edge_no]
            right_edge = self[edge_no + 1:self.shape[0] - 1 - edge_no, self.shape[1] - 1 - edge_no]

            edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

        # noinspection PyAttributeOutsideInit
        self.sky_background_level, self.sky_background_noise = norm.fit(edges)

    def circle_mask(self, radius_arc):
        """
        Create a new circular mask for this image

        Parameters
        ----------
        radius_arc : float
            The radius of the mask

        Returns
        -------
        A circular mask for this image
        """
        return Mask.circular(arc_second_dimensions=self.shape_arc_seconds, pixel_scale=self.pixel_scale,
                             radius=radius_arc)

    def annulus_mask(self, inner_radius_arc, outer_radius_arc):
        """
        Create a new annular mask for this image

        Parameters
        ----------
        inner_radius_arc : float
            The inner radius of the annular mask
        outer_radius_arc : float
            The outer radius of the annular mask

        Returns
        -------
        An annulus mask for this image
        """
        return Mask.annular(arc_second_dimensions=self.shape_arc_seconds, pixel_scale=self.pixel_scale,
                            outer_radius=outer_radius_arc,
                            inner_radius=inner_radius_arc)


def normalize(array):
    return np.divide(array, np.sum(array))


# noinspection PyClassHasNoInit
class Array(np.ndarray):
    """An abstract Array class used for instantiating simple array classes from file"""

    # TODO: Using class methods like this allows us to make the methods create an instance of whichever class they were
    # TODO: called using (e.g. PSF.from_fits -> instance of PSF)
    @classmethod
    def from_fits(cls, filename, hdu, renormalize=True, path=data_path):
        """
        Load an instance from a fits file
        Parameters
        ----------
        filename: String
            The file name
        hdu: Int
            The HDU
        renormalize: Bool
            If true the array will be normalized
        path: String
            The path to the data folder

        Returns
        -------
            A child of the Array class
        """
        array = numpy_array_from_fits(path + filename, hdu)
        return cls.from_array(array, renormalize=renormalize)

    @classmethod
    def from_array(cls, array, renormalize=True):
        if renormalize:
            normalize(array)
        return array.view(cls)


# noinspection PyClassHasNoInit
class PSF(Array):
    pass


# noinspection PyClassHasNoInit
class Noise(Array):
    pass


class Mask(object):
    """Abstract Class for preparing and storing the image mask used for the AutoLens analysis"""

    # TODO: By having this function take a function that decides whether a pixel is part of the mask we can avoid
    # TODO: repeating loops
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
        pixel_dimensions = arc_second_dimensions_to_pixel(arc_second_dimensions, pixel_scale)
        print(pixel_dimensions)
        array = np.zeros((int(pixel_dimensions[0]), int(pixel_dimensions[1])))

        central_pixel_coords = central_pixel(pixel_dimensions)
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
