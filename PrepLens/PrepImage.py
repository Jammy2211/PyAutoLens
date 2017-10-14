from Tools import ImageTools

class PrepImage(object):
    """Class for loading, preparing and manipulating imaging data ready for input into AutoLens.

    Attributes
    ----------
    data2d : ndarray
        Two-dimensional array holding the image data
    xy_dim : list(int)
        x and y dimensions of image (xy_dim[0] = x dimension, xy_dim[1] = y dimension)
    pixel_scale : float
        Size of each pixel in arcseconds
    xy_arcsec : list(float)
        x and y dimensions of image in arcseconds (xy_arcsec[0] = x dimension, xy_arcsec[1] = y dimension)
    """

    def __init__(self, workdir, file, hdu, pixel_scale, sky_estimate_no_edges=3):
        """ Initialize the *PrepImage* class, loading the data and setting up its dimensions

        Parameters
        ----------
        workdir : str
            Directory holding the image data.
        file : str
            The name of the fits file for data to be loaded from.
        hdu : int
            Fits hdu number image is stored in within the fits file.
        pixel_scale : float
            Size of each pixel in arcseconds

        Examples
        --------
        image = PrepImage(workdir='/path/to/image/', file='image1.fits', hdu=0, pixel_scale = 0.1)
        """
        self.data2d, self.xy_dim = ImageTools.load_fits(workdir, file, hdu)
        self.pixel_scale = pixel_scale
        self.xy_arcsec = ImageTools.get_dimensions_arcsec(xy_dim=self.xy_dim, pixel_scale=pixel_scale)
        self.set_sky_estimate(no_edges=sky_estimate_no_edges)
        
        return

    def set_mask_circular(self, mask_radius_arcsec):
        """Setup the image mask, using a circular

        Parameters
        ----------
        mask_radius_arcsec : float
            Circular radius of mask to be generated in arcseconds.

        Examples
        ----------
        image = PrepImage(workdir='/path/to/image/', file='image1.fits', hdu=0, pixel_scale = 0.1)
        image.setup_mask_circular(mask_radius_arcsec=1.6)

        """

        self.mask2d = ImageTools.get_mask_circular(xy_dim=self.xy_dim, pixel_scale=self.pixel_scale,
                                                   mask_radius_arcsec=mask_radius_arcsec)
        
    def set_sky_estimate(self, no_edges):
        """Setup estimate of the background sky level and noise.
        
        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the backgroundd sky properties
        
        Examples
        ----------
        image = PrepImage(workdir='/path/to/image/', file='image1.fits', hdu=0, pixel_scale = 0.1)
        image.setup_sky_estimate(no_edges=3)

        """

        self.sky_estimate, self.noise_estimate = \
            ImageTools.estimate_sky_via_edges(image=self.data2d, no_edges=no_edges)