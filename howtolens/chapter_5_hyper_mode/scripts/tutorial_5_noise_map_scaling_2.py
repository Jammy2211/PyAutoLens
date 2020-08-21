# %%
"""
Tutorial 5: Noise Map Scaling 2
===============================

Noise-map scaling is important when our mass model lead to an inaccurate source reconstruction . However, it serves an
even more important use, when another component of our lens model doesn't fit the data well. Can you think what it is?
What could leave significant residuals in our model-fit? What might happen to also be the highest S/N values in our
image, meaning these residuals contribute *even more* to the chi-squared distribution?

Yep, you guessed it, it's the lens galaxy _LightProfile_ fit and subtraction. Just like our overly simplified mass
profile's mean we can't perfectly reconstruct the source's light, the same is true of the Sersic profiles we use to
fit the lens galaxy's light. Lets take a look.
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

 - The lens galaxy's light is an _EllipticalSersic_.
 - The lens galaxy's _MassProfile_ is an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is an _EllipticalSersic_.
"""

# %%
from howtolens.simulators.chapter_5 import lens_sersic_sie__source_sersic

dataset_type = "chapter_5"
dataset_name = "lens_sersic_sie__source_sersic"
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
Again, we'll use a convenience function to fit the lens data we simulated above.
"""

# %%
def fit_masked_imaging_with_lens_and_source_galaxy(
    masked_imaging, lens_galaxy, source_galaxy
):

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


# %%
"""
Now, lets use this function to fit the lens data. We'll use a lens model with the correct mass model but an incorrect 
lens _LightProfile_. The source will use a magnification based grid.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.05),
        intensity=0.4,
        effective_radius=0.8,
        sersic_index=3.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
    ),
)


source_magnification = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=3.3),
)

fit = fit_masked_imaging_with_lens_and_source_galaxy(
    masked_imaging=masked_imaging,
    lens_galaxy=lens_galaxy,
    source_galaxy=source_magnification,
)

print("Evidence using baseline variances = ", fit.log_evidence)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

# %%
"""
Okay, so its clear that our poor lens light subtraction leaves residuals in the lens galaxy's centre. These pixels 
are extremely high S/N, so they contribute large chi-squared values. For a real strong lens, we could not fit these 
residual features using a more complex _LightProfile_. These types of residuals are extremely common and they are 
caused by nasty, irregular morphological structures in the lens galaxy; nuclear star emission, nuclear rings, bars, etc.

This skewed chi-squared distribution will cause all the same problems we discussed in the previous tutorial, like 
over-fitting. However, for the source-reconstruction and Bayesian log evidence the residuals are even more problematic 
than before. Why? Because when we compute the Bayesian log evidence for the source-inversion these pixels are included 
like all the other image pixels. But, __they do not contain the source__. The Bayesian log evidence is going to try 
improve the fit to these pixels by reducing the level of regularization,  but its __going to fail miserably__, as they 
map nowhere near the source!

This is a fundamental problem when simultaneously modeling the lens galaxy's light and source galaxy. The source 
inversion has no way to distinguish whether the flux it is reconstructing belongs to the lens or source. This is 
why contribution maps are so valuable; by creating a contribution map for every galaxy in the image __PyAutoLens__ has a 
means by which to distinguish which flux belongs to each component in the image! This is further aided by the 
_Pixelization_'s / regularizations that adapt to the source morphology, as not only are they adapting to where the 
source __is*__ they adapt to where __it isn't__ (and therefore where the lens galaxy is).

Lets now create our hyper-galaxy-images and use them create the contribution maps of our lens and source galaxies. 
Note below that we now create separate model images for our lens and source galaxies. This allows us to create 
contribution maps for each.
"""

# %%
hyper_image = fit.model_image.in_1d

hyper_image_lens = fit.model_images_of_planes[0]  # This is the model image of the lens

hyper_image_source = fit.model_images_of_planes[
    1
]  # This is the model image of the source

lens_galaxy_hyper = al.Galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.05),
        intensity=0.4,
        effective_radius=0.8,
        sersic_index=3.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
    ),
    hyper_galaxy=al.HyperGalaxy(
        contribution_factor=0.3, noise_factor=4.0, noise_power=1.5
    ),
    hyper_model_image=hyper_image,
    hyper_galaxy_image=hyper_image_lens,  # <- The lens get its own hyper-galaxy image.
)

source_magnification_hyper = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=3.3),
    hyper_galaxy=al.HyperGalaxy(
        contribution_factor=2.0, noise_factor=2.0, noise_power=3.0
    ),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image_source,  # <- The source get its own hyper-galaxy image.
)

fit = fit_masked_imaging_with_lens_and_source_galaxy(
    masked_imaging=masked_imaging,
    lens_galaxy=lens_galaxy,
    source_galaxy=source_magnification,
)

lens_contribution_map = lens_galaxy_hyper.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image_lens
)

aplt.Array(
    array=lens_contribution_map,
    mask=mask,
    plotter=aplt.Plotter(labels=aplt.Labels(title="Lens Contribution Map")),
)

source_contribution_map = source_magnification_hyper.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image_source
)

aplt.Array(
    array=source_contribution_map,
    mask=mask,
    plotter=aplt.Plotter(labels=aplt.Labels(title="Source Contribution Map")),
)

# %%
"""
The contribution maps decomposes the image into its different components. Next, we  use each contribution map to 
scale different regions of the noise-map. From the fit above it was clear that both the lens and source required the 
noise to be scaled, but their different chi-squared values ( > 150 and ~ 30) means they require different levels of 
noise-scaling. Lets see how much our fit improves and Bayesian log evidence increases.
"""

# %%
fit = fit_masked_imaging_with_lens_and_source_galaxy(
    masked_imaging=masked_imaging,
    lens_galaxy=lens_galaxy_hyper,
    source_galaxy=source_magnification_hyper,
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

print("Evidence using baseline variances = ", 3356.88)

print("Evidence using hyper-galaxy hyper variances = ", fit.log_evidence)


# %%
"""
Great, and with that, we've covered hyper galaxies. You might be wondering, what happens if there are multiple lens 
galaxies? or multiple source galaxies? Well, as you'd expect, __PyAutoLens__ will make each a hyper-galaxy and 
therefore scale the noise-map of that individual galaxy in the image. This is what we want, as different parts of 
the image require different levels of noise-map scaling.

Finally, I want to quickly mention two more ways that we change our data during th fitting process. One scales the 
background noise and one scales the image's background sky. To do this, we use the 'hyper_data' module in __PyAutoLens__.
"""

# %%
"""
This module includes all components of the model that scale parts of the data. To scale the background sky in the 
image we use the HyperImageSky class and input a 'sky_scale'.
"""

# %%
hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)

# %%
"""
The sky_scale is literally just a constant value we add to every pixel of the observed image before fitting it 
therefore increasing or decreasing the background sky level in the image .This means we can account for an 
inaccurate background sky subtraction in our data reduction during __PyAutoLens__ model fitting.

We can also scale the background noise in an analogous fashion, using the HyperBackgroundNoise class and the 
'noise_scale' hyper-galaxy-parameter. This value is added to every pixel in the noise-map.
"""

# %%
hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

# %%
"""
To use these hyper-galaxy-instrument parameters, we pass them to a lens-fit just like we do our _Tracer_.
"""

# %%
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy_hyper, source_magnification_hyper]
)

al.FitImaging(
    masked_imaging=masked_imaging,
    tracer=tracer,
    hyper_image_sky=hyper_image_sky,
    hyper_background_noise=hyper_background_noise,
)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

# %%
"""
Is there any reason to scale the background noise other than if the background sky subtraction has a large 
correction? There is. Lots of pixels in an image do not contain the lensed source but are fitted by the 
inversion. As we've learnt in this chapter, this isn't problematic when we have our adaptive _Regularization_ scheme 
because the regularization_coefficient will be increased to large values.

However, if you ran a full __PyAutoLens__ analysis in hyper-galaxy-mode (which we cover in the next tutorial), you'd 
find the method still dedicates a lot of source-pixels to fit these regions of the image, 
__even though they have no source__. Why is this? Its because although these pixels have no source, they still 
have a relatively high S/N values (of order 5-10) due to the lens galaxy (e.g. its flux before it is subtracted). The 
inversion when reconstructing the data 'sees' pixels with a S/N > 1 and therefore wants to fit them with a high 
resolution.

By increasing the background noise these pixels will go to much lower S/N values (<  1). The adaptive _Pixelization_will 
feel no need to fit them properly and begin to fit these regions of the source-plane with far fewer, much bigger 
source pixels! This will again give us a net increase in Bayesian log evidence, but more importantly, it will dramatically 
reduce the number of source pixels we use to fit the data. And what does fewer source-pixels mean? Much, much faster
run times. Yay!

With that, we have introduced every feature of hyper-galaxy-mode. The only thing left for us to do is to bring it 
all together and consider how we use all of these features in __PyAutoLens__ pipelines. That is what we'll discuss in the 
next tutorial, and then you'll be ready to perform your own hyper-galaxy-fits!
"""
