import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens with two lens galaxies using  two _EllipticalSersic_ _LightProfile_'s, 
two _EllipticalIsothermal_ _MassProfile_'s and a parametric _EllipticalSersic_ source.

The pipeline assumes the lens galaxies are at (0.0", -1.0") and (0.0", 1.0") and is not a general pipeline
and cannot be applied to any image of a strong lens.

The pipeline is four phases:

Phase 1:

    Fit the _LightProfile_ of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the _LightProfile_ of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").
    
    Lens Light: EllipticalSersic + EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the left lens subtracted image from phase 1.

Phase 3:

    Use this lens-subtracted image to fit the source-galaxy's light. The _MassProfile_'s of the two lens galaxies
    can use the results of phases 1 and 2 to initialize their priors.

    Lens Light: EllipticalSersic + EllipticalSersic
    Lens Mass: EllipticalIsothermal + EllipticalIsothermal
    Source Light: EllipticalSersic
    Prior Passing: Lens light (instance -> phases 1 & 2).
    Notes: None
    
Phase 4:

    Fit all relevant parameters simultaneously, using priors from phases 1, 2 and 3.
    
    Lens Light: EllipticalSersic + EllipticalSersic
    Lens Mass: EllipticalIsothermal + EllipticalIsothermal
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phases 1 & 2), Lens mass & Source light (model -> phase 3).
    Notes: None
    
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__x2_galaxies"

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1: Fit the left lens galaxy's light, where we:

        1) Fix the centres to (0.0, -1.0), the pixel we know the left _Galaxy_'s light centre peaks.
    """

    left_lens = al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)
    left_lens.light.centre_0 = 0.0
    left_lens.light.centre_1 = -1.0

    phase1 = al.PhaseImaging(
        phase_name="phase_1__left_lens_light",
        folders=setup.folders,
        galaxies=dict(
            left_lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=30, evidence_tolerance=5.0),
    )

    """
    Phase 2: Fit the lens galaxy on the right, where we:

        1) Fix the centres to (0.0, 1.0), the pixel we know the right _Galaxy_'s light centre peaks.
        2) Pass the left lens's light model as an instance, to improve the fitting of the right galaxy.
    """

    right_lens = al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)
    right_lens.light.centre_0 = 0.0
    right_lens.light.centre_1 = 1.0

    phase2 = al.PhaseImaging(
        phase_name="phase_2__right_lens_light",
        folders=setup.folders,
        galaxies=dict(
            left_lens=phase1.result.instance.galaxies.left_lens, right_lens=right_lens
        ),
        search=af.DynestyStatic(n_live_points=30, evidence_tolerance=5.0),
    )

    """
    Phase 3: Fit the source galaxy, where we: 
    
        1) Perform the lens light subtraction using the models inferred in phases 1 and 2.
        2) Fix the centres of the mass profiles to (0.0, 1.0) and (0.0, -1.0).
        
    Note how when we construct the _GalaxyModel_ we are using the results above to set up the light profiles, but
    using new mass profiles to set up the mass modeling.
    """

    left_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase1.result.instance.galaxies.left_lens.light,
        mass=al.mp.EllipticalIsothermal,
    )

    right_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase2.result.instance.galaxies.right_lens.light,
        mass=al.mp.EllipticalIsothermal,
    )

    left_lens.mass.centre_0 = 0.0
    left_lens.mass.centre_1 = -1.0
    right_lens.mass.centre_0 = 0.0
    right_lens.mass.centre_1 = 1.0

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_x2_sie__source_exp",
        folders=setup.folders,
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential),
        ),
        search=af.DynestyStatic(n_live_points=50, evidence_tolerance=5.0),
    )

    """
    Phase 4: Fit both lens galaxy's light and mass profiles, as well as the source-galaxy, simultaneously, where we:
    
        1) Use the results of phases 1 and 2 to initialize the lens light models.
        2) Use the results of phase 3 to initialize the lens mass and source light models.

    Remember that in the above phases, we fixed the centres of the light and mass profiles. Thus, if we were to simply
    setup these model components using the command:
    
        light=phase1.result.model.galaxies.left_lens.light

    The model would be set up with these fixed centres! We want to treat the centres as free parameters in this phase,
    requiring us to unpack the prior passing and setup the models using a PriorModel.
    
    """

    left_light = af.PriorModel(al.lp.EllipticalSersic)
    left_light.elliptical_comps = (
        phase1.result.model.galaxies.left_lens.light.elliptical_comps
    )
    left_light.intensity = phase1.result.model.galaxies.left_lens.light.intensity
    left_light.effective_radius = (
        phase1.result.model.galaxies.left_lens.light.effective_radius
    )
    left_light.sersic_index = phase1.result.model.galaxies.left_lens.light.sersic_index

    left_mass = af.PriorModel(al.mp.EllipticalIsothermal)
    left_mass.elliptical_comps = (
        phase3.result.model.galaxies.left_lens.mass.elliptical_comps
    )
    left_mass.einstein_radius = (
        phase3.result.model.galaxies.left_lens.mass.einstein_radius
    )

    left_lens = al.GalaxyModel(redshift=0.5, light=left_light, mass=left_mass)

    right_light = af.PriorModel(al.lp.EllipticalSersic)
    right_light.elliptical_comps = (
        phase2.result.model.galaxies.right_lens.light.elliptical_comps
    )
    right_light.intensity = phase2.result.model.galaxies.right_lens.light.intensity
    right_light.effective_radius = (
        phase2.result.model.galaxies.right_lens.light.effective_radius
    )
    right_light.sersic_index = (
        phase2.result.model.galaxies.right_lens.light.sersic_index
    )

    right_mass = af.PriorModel(al.mp.EllipticalIsothermal)
    right_mass.elliptical_comps = (
        phase3.result.model.galaxies.right_lens.mass.elliptical_comps
    )
    right_mass.einstein_radius = (
        phase3.result.model.galaxies.right_lens.mass.einstein_radius
    )

    right_lens = al.GalaxyModel(redshift=0.5, light=right_light, mass=right_mass)

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_x2_sersic_sie__source_exp",
        folders=setup.folders,
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=phase3.result.model.galaxies.source,
        ),
        search=af.DynestyStatic(n_live_points=60, evidence_tolerance=0.3),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
