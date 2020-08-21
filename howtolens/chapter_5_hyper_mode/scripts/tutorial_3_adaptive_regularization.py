# %%
"""
Tutorial 3: Adaptive Regularization
===================================

In tutorial 1, we considered why our _Constant_ _Regularization_scheme was sub-optimal. Diffferent regions of the
source demand different levels of regularization, motivating a _Regularization_scheme which adapts to the reconstructed
source's surface brightness.

This raises the same question as before, how do we adapt our _Regularization_scheme to the source before we've
reconstructed it? Just like in the last tutorial, we'll use a model image of a strongly lensed source from a previous
phase of the pipeline that we've begun calling the 'hyper-galaxy-image'.
"""

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""
We'll use the same strong lensing data as the previous tutorial, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's _MassProfile_ is an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is an _EllipticalSersic_.
"""

# %%
from howtolens.simulators.chapter_5 import lens_sie__source_sersic

dataset_type = "chapter_5"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/howtolens/dataset/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=2, radius=3.0
)

masked_imaging = al.MaskedImaging(
    imaging=imaging,
    mask=mask,
    settings=al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2),
)

# %%
"""
Next, we're going to fit the image using our magnification based grid. To perform the fits, we'll use a convenience 
function to fit the lens data we simulated above.
"""

# %%
def fit_masked_imaging_with_source_galaxy(masked_imaging, source_galaxy):

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


# %%
"""
Next, we'll use the magnification based source to fit this data.
"""

# %%
source_magnification = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=3.3),
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_magnification
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

aplt.Inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# %%
"""
Okay, so the inversion's fit looks just like it did in the previous tutorials. Lets quickly remind ourselves that 
the effective regularization_coefficient of each source pixel is our input coefficient value of 3.3.
"""

# %%
aplt.Inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# %%
"""
Lets now look at adaptive _Regularization_in action, by setting up a hyper-galaxy-image and using the 
'AdaptiveBrightness' _Regularization_scheme. This introduces additional hyper-galaxy-parameters, that I'll explain next.
"""

# %%
hyper_image = fit.model_image.in_1d_binned

source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.005, outer_coefficient=1.9, signal_scale=3.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)

aplt.Inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.Inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# %%
"""
So, as expected, we now have a variable _Regularization_ scheme. The _Regularization_of the source's brightest regions 
is much lower than that of its outer regions. As discussed before, this is what we want. Lets quickly check that this 
does, indeed, increase the Bayesian log evidence:
"""

# %%
print("Evidence using constant _Regularization_= ", 4216)
print("Evidence using adaptive _Regularization_= ", fit.log_evidence)

# %%
"""
Yep! Of course, combining the adaptive _Pixelization_and _Regularization_will only further benefit our lens modeling!

However, as shown below, we don't fit the source as well as the morphology based _Pixelization_ did in the last chapter. 
This is because although the adaptive _Regularization_ scheme improves the fit, the magnification based 
_Pixelization_ simply *does not*  have sufficient resolution to resolve the source's cuspy central _LightProfile_.
"""

# %%
aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

# %%
"""
__How does adaptive regularization work?__

For every source-pixel, we have a mapping between that pixel and a set of pixels in the hyper-galaxy-image. Therefore, 
for every source-pixel, if we sum the values of all hyper-galaxy-image pixels that map to it we get an estimate of 
how much of the lensed source's signal we expect will be reconstructed. We call this each pixel's 'pixel signal'.

If a source-pixel has a higher pixel-signal, we anticipate that it'll reconstruct more flux and we use this information 
to regularize it less. Conversely, if the pixel-signal is close to zero, the source pixel will reconstruct near-zero 
flux and _Regularization_will smooth over these pixels by using a high regularization_coefficient.

This works as follows:

 1) For every source pixel, compute its pixel-signal, the summed flux of all corresponding image-pixels in the 
 hyper-galaxy-image.
    
 2) Divide every pixel-signal by the number of image-pixels that map directly to that source-pixel. In doing so, all 
 pixel-signals are 'relative'. This means that source-pixels which by chance map to more image-pixels than their 
 neighbors will not have a higher pixel-signal, and visa versa. This ensures the specific _Pixelization_
 does impact the adaptive _Regularization_ pattern.
    
 3) Divide the pixel-signals by the maximum pixel signal so that they range between 0.0 and 1.0.
    
 4) Raise these values to the power of the hyper-galaxy-parameter *signal_scale*. For a *signal_scale* of 0.0, all 
 pixels will therefore have the same final pixel-scale. As the *signal_scale* increases, a sharper transition of 
 pixel-signal values arises between regions with high and low pixel-signals.
    
 5) Compute every source pixel's effective regularization_coefficient as:
    
 (inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)) ** 2.0
    
 This uses two regularization_coefficients, one which is applied to pixels with high pixel-signals and one to 
 pixels with low pixel-signals. Thus, pixels in the inner regions of the source may be given a lower level of 
 _Regularization_than pixels further away, as desired.

Thus, we now adapt our _Regularization_ scheme to the source's surface brightness. Where its brighter (and therefore 
has a steeper flux gradient) we apply a lower level of _Regularization_ than further out. Furthermore, in the edges of 
the source-plane where no source-flux is present we will assume a high regularization_coefficients that smooth over 
all source-pixels.

Try looking at a couple of extra solutions which use with different inner and outer regularization_coefficients or 
signal scales. I doubt you'll notice a lot change visually, but the log evidence certainly has a lot of room for 
manoveur with different values.

You may find solutions that raise an 'InversionException'. These solutions mean that the matrix used during the 
linear algebra calculation was ill-defined, and could not be inverted. These solutions are removed by __PyAutoLens__ 
during lens modeling.
"""

# %%
source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)

aplt.Inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.Inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

print("Evidence using adaptive _Regularization_= ", fit.log_evidence)

# %%
"""
To end, lets consider what this adaptive _Regularization_scheme means in the context of maximizing the Bayesian
log_evidence. In the previous tutorial, we noted that by using a brightness-based adaptive _Pixelization_ we increased 
the Bayesian log evidence by allowing for new solutions which fit the data user fewer source pixels; the key criteria 
in making a source reconstruction 'more simple' and 'less complex'.

As you might of guessed, adaptive _Regularization_again increases the Bayesian log evidence by making the source 
reconstruction simpler:

 1) Reducing _Regularization_ in the source's brightest regions produces a 'simpler' solution in that we are not 
 over-smoothing our reconstruction of its brightest regions.
    
 2) Increasing _Regularization_ in the outskirts produces a simpler solution by correlating more source-pixels, 
 effectively reducing the number of pixels used by the reconstruction.

Together, brightness based _Pixelization_'s and _Regularization_ allow us to find the objectively 'simplest' source 
solution possible and therefore ensure that our Bayesian log evidence's have a well defined maximum value they are 
seeking. This was not the case for magnification based _Pixelization_'s and constant _Regularization_ schemes.
"""
