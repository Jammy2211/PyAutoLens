import autolens as al

from test_autolens.simulate.imaging import simulate_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = simulate_util.load_test_imaging(
    data_type="lens_sis__source_smooth", data_resolution="hst"
)
mask = al.mask.circular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    radius=3.0,
    centre=(0.0, 0.0),
)

lens_galaxy = al.Galaxy(
    redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

aplt.fit_imaging.subplot_of_plane(
    fit=fit,
    plane_index=1,
    plot_in_kpc=True,
    include_image_plane_pix=True,
    include_caustics=True,
)

aplt.fit_imaging.subplot_of_plane(
    fit=fit,
    plane_index=1,
    plot_in_kpc=False,
    include_image_plane_pix=True,
    include_caustics=True,
)


aplt.inversion.subplot_imaging(inversion=fit.inversion)
