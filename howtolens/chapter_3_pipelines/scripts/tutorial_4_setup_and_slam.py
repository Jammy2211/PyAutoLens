# %%
"""
Tutorial 4: Setup and SLaM
==========================

You are now familiar with pipelines, in particular how we use them to break-down the lens modeling procedure
to provide more efficient and reliable model-fits. In the previous tutorials, you learnt how to write your own
pipelines, which can fit whatever lens model is of particular interest to your scientific study.

However, for most lens models there are standardized approaches one can take to fitting them. For example, as we saw in
tutorial 1 of this chapter, an effective approach is to fit a model for the lens's light followed by a model for its
mass and the source. It would be wasteful for all **PyAutoLens** users to have to write their own pipelines to
perform the same tasks.

For this reason, the `autolens_workspace` comes with a number of standardized pipelines, which fit common lens models
in ways we have tested are efficient and robust. These pipelines also use `Setup` objects to customize the creating of
the lens and source `PriorModel`'s, making it straight forward to use the same pipeline to fit a range of different
lens model parameterizations.

Lets take a look.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al

# %%
"""
Lets begin with the `SetupLightParametric` object, which describes how we set up our parametric model using 
`LightProfile`'s for the lens's light (we call it parametric to make it clear the model uses `LightProfile`'s to fit
the lens's light, in contrast to the `Inversion` objects introduced in the next chapter of **HowToLens**).

This object customizes:

 - The `LightProfile`'s which fit different components of the lens light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 - If the centre of the lens light profile is manually input and fixed for modeling.
"""

# %%
setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=al.lp.EllipticalSersic,
    envelope_prior_model=None,
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

# %%
"""
In the `Setup` above we made the lens's `bulge` and `disk` use the `EllipticalSersic` `LightProfile`, which we
can verify below:
"""

# %%
print(setup_light.bulge_prior_model)
print(setup_light.bulge_prior_model.cls)
print(setup_light.disk_prior_model)
print(setup_light.disk_prior_model.cls)

# %%
"""
We can also verify that the `bulge` and `disk` share the same prior on the centre because we aligned them
by setting `align_bulge_disk_centre=True`:
"""

# %%
print(setup_light.bulge_prior_model.centre)
print(setup_light.disk_prior_model.centre)

# %%
"""
When `GalaxyModel`'s are created in the template pipelines in the `autolens_workspace/transdimensional/pipelines`
and `autolens_workspace/slam/pipelines` they use the `bulge_prior_model`, `disk_prior_model`, etc to create them (as 
opposed to explicitly writing the classes in the pipelines, as we did in the previous tutorials).
"""

# %%
"""
The `SetupSourceParametric` object works in exactly the same way as the `SetupLightParametric` object above, but 
instead for choosing the source model. This object makes it straight forward to fit parametric source models with many
different numbers of components (e.g. just a bulge, a bulge and disk, etc).

All inputs of `SetupSourceParametric` are identical to `SetupLightParametric`.
"""

# %%
setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalCoreSersic,
    disk_prior_model=al.lp.EllipticalExponential,
    envelope_prior_model=None,
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

print(setup_source.bulge_prior_model)
print(setup_source.bulge_prior_model.cls)
print(setup_source.disk_prior_model)
print(setup_source.disk_prior_model.cls)

# %%
"""
The lens mass model is customized using the `SetupMassTotal` object, which customizes:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not (this lens was not simulated with shear and 
   we do not include it in the mass model).
 - If the centre of the lens mass profile is manually input and fixed for modeling.
"""

# %%
setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=False, mass_centre=None
)

# %%
"""
There is also a `SetupMassLightDark` object, which customizes lens mass models which decompose the lens's mass
distribution into stellar and dark matter. More information on these models can be found in the 
`autolens_workspace/examples` and `autolens_workspace/transdimensional` folders.
"""

# %%
"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `with_shear` 
is True, the pipeline`s output paths are `tagged` with the string `with_shear`.

This means you can run the same pipeline on the same data twice (e.g. with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `path_prefix` specifies the path the pipeline results are written to, as it did with phases in the previous 
chapter. The redshift of the lens and source galaxies are also input.
"""

# %%
setup = al.SetupPipeline(
    path_prefix=f"path_prefix",
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

# %%
"""
_Template Pipelines_

The template pipelines can be found in the folder `autolens_workspace/transdimensional/pipelines`, with their 
accompanying runner scripts in `autolens_workspace/transdimensional/runners`. These templates are pretty comprehensive, 
so we do not include any pipelines in this tutorial of **HowToLens**, you should check out the `transdimensional` 
package now!
"""

# %%
"""
__SLaM (Source, Light and Mass)__

A second set of template pipelines, called the **SLaM** (Source, Light and Mass) pipelines can be found in the folder
`autolens_workspace/slam`. These are similar in design to the `transdimensional` pipelines, but are composed of 
the following specific pipelines:

 - `Source`: A pipeline that focuses on producing a robust model for the source's light, using simpler models for the 
   lens's light (e.g. a `bulge` + `disk`) and mass (e.g. an `EllipticalSersic`). 
   
 - `Light`: A pipeline that fits a complex lens light model (e.g. one with many components), using the initialized 
   source model to cleanly deblend the lens and source light.
   
 - `Mass`: A pipeline that fits a complex lens mass model, benefitting from the good models for the lens's light and 
   source.

For fitting very complex lens models, for example ones which decompose its mass into its stellar and dark components,
the **SLaM** pipelines have been carefully crafted to do this in a reliable and automated way that is still efficient. 

The **SLaM** pipelines also make fitting many different models to a single dataset efficient, as they reuse the results 
of earlier phases (e.g. in the Source pipeline) to fit different models in the `Light` and `Mass` pipelines for the 
lens's  light and mass.
"""

# %%
"""
Whether you should use phases, `transdimensional` pipelines, `slam` pipelines or write your own depends on the scope 
of your scientific analysis. I would advise you begin by adapting the scripts in `autolens/examples` to fit your
data, and then do so using the `transdimensional` or `slam` pipelines once things seem to be working well!
"""
