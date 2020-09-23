# %%
"""
Tutorial 8: Pipeline
====================

To illustrate lens modeling using an `Inversion` and `Pipeline`, we'll go back to the complex source model-fit that we
performed in tutorial 3 of chapter 3. This time, as you`ve probably guessed, we'll fit the complex source using an
_Inversion_.

we'll begin by modeling the source with a `LightProfile`, to initialize the mass model and avoid the unphysical
solutions discussed in tutorial 6. we'll then switch to an `Inversion`.
"""

""" AUTOFIT + CONFIG SETUP """

# %%
#%matplotlib inline

from autoconf import conf
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
Use this path to explicitly set the config path and output path.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtolens/config",
    output_path=f"{workspace_path}/howtolens/output",
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autolens as al
import autolens.plot as aplt

# %%
"""
we'll use strong lensing data, where:

 - The lens galaxy`s light is omitted.
 - The lens galaxy`s `MassProfile` is an `EllipticalIsothermal`.
 - The source galaxy`s `LightProfile` is four `EllipticalSersic``..
"""

# %%
from howtolens.simulators.chapter_4 import mass_sie__source_sersic_x4

dataset_type = "chapter_4"
dataset_name = "mass_sie__source_sersic_x4"
dataset_path = f"{workspace_path}/howtolens/dataset/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
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

For this pipeline the pipeline setup customizes and tags:

 - If there is an `ExternalShear` in the mass model or not.
 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by of this pipeline.
"""

# %%
setup_mass = al.SetupMassTotal(no_shear=False)
setup_source = al.SetupSourceInversion(
    pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
)

setup = al.SetupPipeline(
    folders=["c4_t8_inversion"], setup_mass=setup_mass, setup_source=setup_source
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
*Setup* and *SettingsPhase* above.
"""

# %%
from howtolens.chapter_4_inversions import tutorial_8_pipeline

pipeline_inversion = tutorial_8_pipeline.make_pipeline(setup=setup, settings=settings)

# Uncomment to run.
# pipeline_inversion.run(dataset=imaging, mask=mask)

# %%
"""
And with that, we now have a pipeline to model strong lenses using an inversion! Checkout the example pipeline in
`autolens_workspace/pipelines/examples/inversion_hyper_galaxies_bg_noise.py` for an example of an `Inversion` pipeline 
that includes the lens light component.
"""
