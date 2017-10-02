from astropy.io import fits
import numpy as np

def load_fits(file, hdu):
    """Load a files and return the image data"""
    hdulist = fits.open(file)  # Open the fits file
    hdulist.info()  # Display fits header info
    data = np.array(hdulist[hdu].data)  # Store as image, which is returned by function
    return data