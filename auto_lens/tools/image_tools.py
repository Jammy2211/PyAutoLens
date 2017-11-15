from astropy.io import fits
import numpy as np


# TODO: Some words are reserved because python uses them. I've changed file to filename here because file is one of
# TODO: words. Also, filename seems like a better description of the variable you're using here.
def load_fits(work_dir, filename, hdu):
    """Load the input image file and return the image data and dimensions

    Parameters
    ----------
    work_dir : str
        Directory holding the image data.
    filename : str
        The name of the fits file for data to be loaded from.
    hdu : int
        Fits hdu number image is stored in within the fits file.

    Returns
    ----------
    data2d : ndarray
        Two-dimensional image read from the fits file
    xy_dim : list(int)
        x and y dimensions of image (xy_dim[0] = x dimension, xy_dim[1] = y dimension)

    Examples
    --------
    data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file=testdir + '3x3_ones.fits', hdu=0)

    """
    hdu_list = fits.open(work_dir + filename)  # Open the fits file
    #   hdu_list.info()  # Display fits header info
    data2d = np.array(hdu_list[hdu].data)  # Store as image, which is returned by function

    xy_dim = data2d.shape[:]  # x dimension (pixels)

    return data2d, xy_dim
