import numpy as np

from auto_lens.imaging import imaging



class RayTracingGrids(object):

    def __init__(self, image, sub=None, blurring=None):

        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def from_mask(cls, mask, sub_grid_size=None, blurring_size=None):

        image = AnalysisGridImage.from_mask(mask)

        if sub_grid_size is None:
            sub = None
        elif sub_grid_size is not None:
            sub = AnalysisGridImageSub.from_mask(mask, sub_grid_size)

        if blurring_size is None:
            blurring = None
        elif blurring_size is not None:
            blurring = AnalysisGridBlurring.from_mask(mask, blurring_size)

        return RayTracingGrids(image, sub, blurring)


class AnalysisGrid(object):

    def __init__(self, grid):
        """Abstract base class for analysis grids, which store the pixel coordinates of different regions of an \
        image and where the ray-tracing and lensing analysis are performed.

        The different regions represented by each analysis grid are used for controlling different aspects of the \
        analysis (e.g. the image, the image sub-grid, the clustering grid, etc.)

        The grids are stored as a structured array of pixel coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds.
        """

        self.grid = grid


class AnalysisGridImage(AnalysisGrid):

    def __init__(self, grid):
        """The image analysis grid, representing all pixel coordinates in an image where the ray-tracing and lensing
        analysis is performed.

        Parameters
        -----------
        grid : np.ndarray[image_pixels, 2]
            Array containing the image grid coordinates. The first elements maps to an image pixel, and second to its \
            (x,y) arc second coordinates. E.g. the value grid[3,1] give the 4th image pixel's y coordinate.
        """

        super(AnalysisGridImage, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the image analysis grid using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image grid is computed for and the image's data grid.
        """
        return AnalysisGridImage(mask.compute_image_grid())


class AnalysisGridImageSub(AnalysisGrid):

    def __init__(self, grid, sub_grid_size):
        """The image analysis sub-grid, representing all pixel sub-coordinates in an image where the ray-tracing and \
        lensing analysis is performed.

        Parameters
        -----------
        grid : np.ndarray[image_pixels, sub_grid_size**2, 2]
            Array containing the sub-grid coordinates. The first elements maps to an image pixel, the second to its \
            sub-pixel and third to its (x,y) arc second coordinates. E.g. the value grid[3,6,1] give the 4th image \
            pixel's 7th sub-pixel's y coordinate.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        self.sub_grid_size = sub_grid_size
        super(AnalysisGridImageSub, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask, sub_grid_size):
        """ Given an image.Mask, compute the image analysis sub-grid using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image sub-grid is computed for and the image's data grid.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        return AnalysisGridImageSub(mask.compute_image_sub_grid(sub_grid_size), sub_grid_size)


class AnalysisGridBlurring(AnalysisGrid):

    def __init__(self, grid):
        """The blurring analysis grid, representing all pixels which are outside the image mask but will have a \
         fraction of their light blurred into the mask via PSF convolution.

        Parameters
        -----------
        grid : np.ndarray[blurring_pixels, 2]
            Array containing the blurring grid coordinates. The first elements map to an image pixel, and second to its \
            (x,y) arc second coordinates. E.g. the value grid[3,1] give the 4th blurring pixel's y coordinate.
        """

        super(AnalysisGridBlurring, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask, psf_size):
        """ Given an image.Mask, compute the blurring analysis grid by locating all pixels which are within the \
        psf size of the mask.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid is computed for and the image's data grid.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """
        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise imaging.MaskException("psf_size of exterior region must be odd")

        return AnalysisGridBlurring(mask.compute_blurring_grid(psf_size))


class AnalysisMapperSparse(object):
    """Analysis grid mappings between the sparse grid and image grid.
    """

    def __init__(self, sparse_to_image, image_to_sparse):

        self.sparse_to_image = sparse_to_image
        self.image_to_sparse = image_to_sparse

    @classmethod
    def from_mask(cls, mask, sparse_grid_size):
        sparse_to_image, image_to_sparse = mask.compute_sparse_mappers(sparse_grid_size)
        return AnalysisMapperSparse(sparse_to_image, image_to_sparse)