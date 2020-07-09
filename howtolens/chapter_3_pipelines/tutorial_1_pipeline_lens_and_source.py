import autofit as af
import autolens as al

"""
All pipelines begin with a comment describing the pipeline and a phase-by-phase description of what it does.

In this pipeline, we fit the a strong lens using an _EllipticalSersic_ _LightProfile_, _EllipticalIsothermal_ 
_MassProfile_ and parametric _EllipticalSersic_ source.

The pipeline is three phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_.
    
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


def make_pipeline(setup, settings):

    pipeline_name = "pipeline__light_and_source"

    """
    A pipelines takes the 'folders' as input, which together with the pipeline name specify the path structure 
    of the output. In the pipeline runner we pass the folders ["howtolens", c3_t1_lens_and_source], making the
    output of this pipeline 'autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline__light_and_source'.

    The output path is also tagged according to the _PipelineSetup_, in an analagous fashion to how the 
    _PhaseSettingsImaging_ tagged the output paths of phases. In this example, we do not use an _ExternalShear_
    in the mass model, and the pipeline is tagged accordingly.
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1: Fit only the lens galaxy's light, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    We create the phase using the same notation as in chapter 2. Note how we are using the 'fast' _Dynesty_ settings
    covered in chapter 2.
    """

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        folders=setup.folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        settings=settings,
        search=af.DynestyStatic(n_live_points=30, evidence_tolerance=5.0),
    )

    """
    Phase 2: Fit the lens galaxy's mass and source galaxy's light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
        2) Set priors on the centre of the lens galaxy's _MassProfile_ by linking them to those inferred for 
           the _LightProfile_ in phase 1.
           
    In phase 2, we fit the source-galaxy's light. Thus, we want to fix the lens light model to the model inferred
    in phase 1, ensuring the image we fit is lens subtracted. We do this below by passing the lens light as an
    'instance' object, a trick we use in nearly all pipelines!

    By passing an 'instance', we are telling __PyAutoLens__ that we want it to pass the maximum log likelihood result of
    that phase and use those parameters as fixed values in the model. The model parameters passed as an 'instance' are
    not free parameters fitted for by the non-linear search, thus this reduces the dimensionality of the non-linear 
    search making model-fitting faster and more reliable. 
     
    Thus, phase2 includes the lens light model from phase 1, but it is completely fixed during the model-fit!
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.light.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.light.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.instance.galaxies.lens.light,
                mass=mass,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, evidence_tolerance=5.0),
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens's light, mass, and source's light using the results of phases 1 and 2.
        
    As in chapter 2, we can use the 'model' attribute to do this. Our _Dynesty_ search now uses slower and more 
    thorough settings than the previous phases, to ensure we robustly quantify the errors.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase2.result.model.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=phase2.result.model.galaxies.source.light
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
