import autolens as al
import autolens.plot as aplt
from test_autolens.simulators.imaging import instrument_util

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses ir'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
imaging = instrument_util.load_test_imaging(
    dataset_name="light_sersic__source_sersic", instrument="vro"
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    radius=2.0,
    centre=(3.0, 3.0),
    sub_size=1,
)

# The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.lmp.EllipticalSersic(
        centre=(0.0, 0.0), intensity=0.1, mass_to_light_ratio=1.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(-0.055555, 0.096225),
        intensity=0.1,
        effective_radius=0.5,
        sersic_index=3.0,
    ),
)

masked_imaging = al.MaskedImaging(
    imaging=imaging,
    mask=mask,
    settings=al.SettingsMaskedImaging(
        grid_class=al.GridInterpolate, pixel_scales_interp=0.1, sub_size=1
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy])
fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(critical_curves=False, caustics=False)
)
