import sys
sys.path.append("../../auto_lens/")

import image

mock_image = image.SimulatedImage(image_dimensions=(100, 100), pixel_scale=0.1)
psf = image.PSF.from_fits(,,
mask = mock_image.unmasked()

mock_psf.plot()