# %%
"""
Tutorial 8: Pipeline
====================

To illustrate lens modeling using an `Inversion` and `Pipeline`, we'll go back to the complex source model-fit that we
performed in tutorial 3 of chapter 3. This time, as you`ve probably guessed, we'll fit the complex source using an
`Inversion`.

we'll begin by modeling the source with a `LightProfile`, to initialize the mass model and avoid the unphysical
solutions discussed in tutorial 6. we'll then switch to an `Inversion`.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

# %%
"""
we'll use strong lensing data, where:

 - The lens `Galaxy`'s light is omitted.
 - The lens `Galaxy`'s total mass distribution is an `EllipticalIsothermal`.
 - The source `Galaxy`'s `LightProfile` is four `EllipticalSersic``..
"""

# %%
dataset_name = "mass_sie__source_sersic_x4"
dataset_path = path.join("dataset", "howtolens", "chapter_4", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)


aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function. We discussed
these in chapter 2, and a full description of all settings can be found in the example script:

 `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline. Note how we can use the _SettingsPixelization_
object to determine whether the border is used during the model-fit.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(sub_size=2)
settings_pixelization = al.SettingsPixelization(use_border=True)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging,
    settings_pixelization=settings_pixelization,
)

# %%
"""
__Pipeline_Setup_And_Tagging__:

We will use the standardized `Setup` objects in this pipeline, which as discussed in chapter 3 provide us with 
covenient and standardized tools to compose a lens model and tags the output paths. 

We saw the `SetupMassTotal` object in the previous chapter, which:

For this pipeline the pipeline setup customizes and tags:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not.
"""

# %%
setup_mass = al.SetupMassTotal(with_shear=True)

# %%
"""
We also use the `SetupSourceInversion` object to customize the `Inversion` used for the source, specifically:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by the `Inversion` of this pipeline.
"""

# %%
setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiMagnification,
    regularization_prior_model=al.reg.Constant,
)

setup = al.SetupPipeline(
    path_prefix=path.join("howtolens", "c4_t8_inversion"),
    setup_mass=setup_mass,
    setup_source=setup_source,
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

# %%
from pipelines import tutorial_8_pipeline

pipeline_inversion = tutorial_8_pipeline.make_pipeline(setup=setup, settings=settings)

# Uncomment to run.
# pipeline_inversion.run(dataset=imaging, mask=mask)

# %%
"""
And with that, we now have a pipeline to model strong lenses using an inversion! Checkout the example pipeline in
`autolens_workspace/pipelines/examples/inversion_hyper_galaxies_bg_noise.py` for an example of an `Inversion` pipeline 
that includes the lens light component.
"""
