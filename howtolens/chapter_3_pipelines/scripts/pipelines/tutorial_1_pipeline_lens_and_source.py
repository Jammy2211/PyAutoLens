import autofit as af
from os import path
import autolens as al

"""
All pipelines begin with a comment describing the pipeline and a phase-by-phase description of what it does.

In this pipeline, we fit the a strong lens using an `EllipticalSersic` `LightProfile`, `EllipticalIsothermal` 
`MassProfile` and parametric `EllipticalSersic` source.

The pipeline is three phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source `LightProfile`.
    
    Lens Light: EllipticalSersic
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None
"""


def make_pipeline(path_prefix, settings, redshift_lens=0.5, redshift_source=1.0):

    pipeline_name = "pipeline__light_and_source"

    """
    A pipelines takes the `path_prefix` as input, which together with the `pipeline_name` specifies the path structure 
    of the output. In the pipeline runner we pass the `path_prefix` f"howtolens/c3_t1_lens_and_source", making the
    output of this pipeline `autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline__light_and_source`.
    """

    """
    Phase 1: Fit only the lens `Galaxy`'s light, where we:

        1) Set priors on the lens galaxy $(y,x)$ centre such that we assume the image is centred around the lens galaxy.

    We create the phase using the same notation as in chapter 2. Note how we are using the `fast` `Dynesty` settings
    covered in chapter 2.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[bulge]", n_live_points=30, evidence_tolerance=5.0
        ),
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, bulge=al.lp.EllipticalSersic)
        ),
        settings=settings,
    )

    """
    Phase 2: Fit the lens`s `MassProfile`'s and source `Galaxy`'s light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
        2) Set priors on the centre of the lens `Galaxy`'s total mass distribution by linking them to those inferred for 
           the `LightProfile` in phase 1.
           
    In phase 2, we fit the source-`Galaxy`'s light. Thus, we want to fix the lens light model to the model inferred
    in phase 1, ensuring the image we fit is lens subtracted. We do this below by passing the lens light as an
 `instance` object, a trick we use in nearly all pipelines!

    By passing an `instance`, we are telling **PyAutoLens** that we want it to pass the maximum log likelihood result of
    that phase and use those parameters as fixed values in the model. The model parameters passed as an `instance` are
    not free parameters fitted for by the non-linear search, thus this reduces the dimensionality of the non-linear 
    search making model-fitting faster and more reliable. 
     
    Thus, phase2 includes the lens light model from phase 1, but it is completely fixed during the model-fit!
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.bulge.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.bulge.centre_1

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_mass[sie]_source[bulge]",
            n_live_points=50,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                mass=mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens`s light, mass, and source`s light using the results of phases 1 and 2.
        
    As in chapter 2, we can use the `model` attribute to do this. Our `Dynesty` search now uses slower and more 
    thorough settings than the previous phases, to ensure we robustly quantify the errors.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[bulge]_mass[sie]_source[bulge]", n_live_points=100
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                mass=phase2.result.model.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                bulge=phase2.result.model.galaxies.source.bulge,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(pipeline_name, path_prefix, phase1, phase2, phase3)
