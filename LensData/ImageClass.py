from LensData import FitsTools, ArrayTools

class Image(object):
    """Class which stores the lensing image data in a 2D array"""

    def __init__(self, file, pixel_scale, imagehdu=1):

        self.file = file # file name
        self.pixel_scale = pixel_scale # arcsecond per pixel size

        self.data = FitsTools.load_fits(file=file, hdu=imagehdu) # Load image from fits file
        self.xd