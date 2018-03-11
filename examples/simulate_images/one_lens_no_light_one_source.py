import sys
sys.path.append("../../auto_lens/")

import image

mock_image = image.SimulatedImage(image_dimensions=(100, 100), pixel_scale=0.1)
psf = image.PSF.from_fits(filename='slacs/slacs_1_post.fits', hdu=3)
mask = image.Mask.off

mock_psf.plot()