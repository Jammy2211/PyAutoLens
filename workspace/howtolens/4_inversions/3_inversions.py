from workspace.howtolens.simulations import inversions as simulate_image

from autolens.imaging import image as im
from autolens.imaging import mask as ma
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_image as li
from autolens.inversion import pixelizations as pix
from autolens.inversion import regularization as reg
from autolens.inversion import inversions as inv
from autolens.plotting import imaging_plotters
from autolens.plotting import inversion_plotters

# So, we've covered mappers, which map source-pixels to an image, and visa versa. Now, we're gonna cover how we can
# use a mapper to reconstruct the source galaxy - I hope you're excited!

# First, lets do all the we did in the previous tutorial. That is, simulate a lens, set up its image, mask, grids,
# galaxies  and tracer and use them to create a pixelization and mapper.
simulate_image.tutorial_2_image()
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/4_inversions'
image = im.load_imaging_from_path(image_path=path + '/data/2_mappers/image.fits',
                                  noise_map_path=path+'/data/2_mappers/noise_map.fits',
                                  psf_path=path + '/data/2_mappers/psf.fits', pixel_scale=0.05)
mask = ma.Mask.annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=1.0,
                       outer_radius_arcsec=2.2)
lensing_image = li.LensingImage(image=image, mask=mask)
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                    einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7, phi=135.0, intensity=0.2,
                                                   effective_radius=0.2, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=lensing_image.grids)
rectangular = pix.Rectangular(shape=(25, 25))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# We're now going to use our mapper to invert the image, using the 'inversions' module, which is imported as 'inv'.
# Before I cover how this works, we'll just go ahead and do the inversion first, it'll be good to see how it looks!
# (To do this, we're also using a 'regularization' - this is covered in the next tutorial, so ignore it for now).
inversion = inv.Inversion(image=lensing_image[:], noise_map=lensing_image.noise_map,
                          convolver=lensing_image.convolver_mapping_matrix, mapper=mapper,
                          regularization=reg.Constant(coeffs=(1.0,)))

inversion_plotters.plot_reconstructed_image(inversion=inversion)

inversion_plotters.plot_reconstructed_pixelization(inversion=inversion)