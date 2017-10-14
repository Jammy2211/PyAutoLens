from astropy.io import fits
import numpy as np

def load_fits(workdir, file, hdu):
    """Load the input image file and return the image data and dimensions

    Parameters
    ----------
    workdir : str
        Directory holding the image data.
    file : str
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
    hdulist = fits.open(workdir+file)  # Open the fits file
    #   hdulist.info()  # Display fits header info
    data2d = np.array(hdulist[hdu].data)  # Store as image, which is returned by function

    xy_dim = data2d.shape[:] # x dimension (pixels)

    return data2d, xy_dim

def get_dimensions_arcsec(xy_dim, pixel_scale):
    """Calculate the image dimensions in arcseconds, from the pixel dimensions and arcsecond pixel scale.

    Parameters
    ----------
    xy_dim : list(int)
        x and y pixel dimensions of image (xy_dim[0] = x dimension, xy_dim[1] = y dimension)
    pixel_scale : float
        Size of each pixel in arcseconds

    Returns
    ----------
    xy_arcsec : list(float)
        x and y dimensions of image in arcseconds (xy_arcsec[0] = x dimension, xy_arcsec[1] = y dimension)

    Example
    ----------

    xy_arcsec = ImageTools.get_dimensions_arcsec(xy_dim=[3,3], pixel_scale=0.1)

    """
    xy_arcsec = list(map(lambda l : l*pixel_scale , xy_dim))
    return xy_arcsec

def get_mask_circular(xy_dim, pixel_scale, mask_radius_arcsec):
    """
    Calculate a circular mask given the image dimensions (pixels), pixel to arcsecond scale and mask radius
    (arcseconds).

    The centre of the mask is assumed to be the centre of the image dimensions (use trim / padding routines to
    recentre the image to the lens if necessary).

    Parameters
    ----------
    xy_dim : list(int)
        x and y pixel dimensions of image (xy_dim[0] = x dimension, xy_dim[1] = y dimension)
    pixel_scale : float
        Size of each pixel in arcseconds
    mask_radius_arcsec : float
        Circular radius of mask to be generated in arcseconds.

    Returns
    ----------
    mask2d : ndarray
        Two-dimensional array of mask, =1 for included in mask, =0 for excluded.

    """

    # Calculate the central pixel of the mask. This is a half pixel value for an even sized array.
    # Also minus one from value so that mask2d is shifted to python array (i.e. starts at 0)
    xy_cen_pix = list(map(lambda l : ((l+1)/2)-1, xy_dim ))

    mask2d = np.zeros((xy_dim[0], xy_dim[1]))

    for i in range(xy_dim[0]):
        for j in range(xy_dim[1]):

            r_arcsec = pixel_scale*np.sqrt((i-xy_cen_pix[0])**2 + (j-xy_cen_pix[1])**2)

            if r_arcsec <= mask_radius_arcsec:
                mask2d[i,j] = int(1)

    return mask2d

def estimate_sky_via_edges(image, no_edges=1):
    """Estimate the background sky level and noise by binning pixels located at the edge(s) of an image into a
    histogram and fitting a Gaussian profile to this histogram. The mean (mu) of this Gaussian gives the background
    sky level, whereas the FWHM (sigma) gives the noise estimate.

    Parameters
    ----------
    image : ndarray
        The two-dimensional imaging data from which the background sky is estimated
    no_edges : int
        Number of edges used to estimate the backgroundd sky properties

    Returns
    ----------
    mu : float
        An estimate of the mean value of background sky in the image
    sigma : float
        An estimate of the standard deviation of the background sky

    """

    from scipy.stats import norm

    xy_dim = image.shape[:] # x dimension (pixels)

    xdim = xy_dim[0]
    ydim = xy_dim[1]

    edges = []

    for edge_no in range(no_edges):

        top_edge    = image[edge_no,                  edge_no:ydim-edge_no]
        bottom_edge = image[xdim-1-edge_no,           edge_no:ydim-edge_no]
        left_edge   = image[edge_no+1:xdim-1-edge_no, edge_no]
        right_edge  = image[edge_no+1:xdim-1-edge_no, ydim-1-edge_no]

        edges = np.concatenate(((edges, top_edge, bottom_edge, right_edge, left_edge)))

    mu, sigma = norm.fit(edges)

    return mu, sigma
