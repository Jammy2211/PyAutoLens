import autofit as af
from os import path
import autolens as al

"""
In this pipeline, we fit the a strong lens with two lens galaxies using  two `EllipticalSersic` `LightProfile`'s, 
two `EllipticalIsothermal` `MassProfile`'s and a parametric `EllipticalSersic` source.

The pipeline assumes the lens galaxies are at (0.0", -1.0") and (0.0", 1.0") and is not a general pipeline
and cannot be applied to any image of a strong lens.

The pipeline is four phases:

Phase 1:

    Fit the `LightProfile` of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the `LightProfile` of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").
    
    Lens Light: EllipticalSersic + EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the left lens subtracted image from phase 1.

Phase 3:

    Use this lens-subtracted image to fit the source-`Galaxy`'s light. The `MassProfile`'s of the two lens galaxies
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


def make_pipeline(path_prefix, settings, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__x2_galaxies"

    """
    Phase 1: Fit the left lens `Galaxy`'s light, where we:

        1) Fix the centres to (0.0, -1.0), the pixel we know the left `Galaxy`'s light centre peaks.
    """

    left_lens = al.GalaxyModel(redshift=redshift_lens, bulge=al.lp.EllipticalSersic)
    left_lens.bulge.centre_0 = 0.0
    left_lens.bulge.centre_1 = -1.0

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]__left_lens_light[bulge]",
            n_live_points=30,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            left_lens=al.GalaxyModel(
                redshift=redshift_lens, bulge=al.lp.EllipticalSersic
            )
        ),
        settings=settings,
    )

    """
    Phase 2: Fit the lens galaxy on the right, where we:

        1) Fix the centres to (0.0, 1.0), the pixel we know the right `Galaxy`'s light centre peaks.
        2) Pass the left lens`s light model as an instance, to improve the fitting of the right galaxy.
    """

    right_lens = al.GalaxyModel(redshift=redshift_lens, bulge=al.lp.EllipticalSersic)
    right_lens.bulge.centre_0 = 0.0
    right_lens.bulge.centre_1 = 1.0

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]__right_lens_light[bulge]",
            n_live_points=30,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            left_lens=phase1.result.instance.galaxies.left_lens, right_lens=right_lens
        ),
    )

    """
    Phase 3: Fit the source galaxy, where we: 
    
        1) Perform the lens light subtraction using the models inferred in phases 1 and 2.
        2) Fix the centres of the mass profiles to (0.0, 1.0) and (0.0, -1.0).
        
    Note how when we construct the `GalaxyModel` we are using the results above to set up the light profiles, but
    using new mass profiles to set up the mass modeling.
    """

    left_lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=phase1.result.instance.galaxies.left_lens.bulge,
        mass=al.mp.EllipticalIsothermal,
    )

    right_lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=phase2.result.instance.galaxies.right_lens.bulge,
        mass=al.mp.EllipticalIsothermal,
    )

    left_lens.mass.centre_0 = 0.0
    left_lens.mass.centre_1 = -1.0
    right_lens.mass.centre_0 = 0.0
    right_lens.mass.centre_1 = 1.0

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]__mass_x2[sie]__source[exp]",
            n_live_points=50,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=al.GalaxyModel(
                redshift=redshift_source, bulge=al.lp.EllipticalExponential
            ),
        ),
    )

    """
    Phase 4: Fit both lens `Galaxy`'s light and mass profiles, as well as the source-galaxy, simultaneously, where we:
    
        1) Use the results of phases 1 and 2 to initialize the lens light models.
        2) Use the results of phase 3 to initialize the lens mass and source light models.

    Remember that in the above phases, we fixed the centres of the light and mass profiles. Thus, if we were to simply
    setup these model components using the command:
    
        bulge=phase1.result.model.galaxies.left_lens.bulge

    The model would be set up with these fixed centres! We want to treat the centres as free parameters in this phase,
    requiring us to unpack the prior passing and setup the models using a PriorModel.
    
    """

    left_sersic = af.PriorModel(al.lp.EllipticalSersic)
    left_sersic.elliptical_comps = (
        phase1.result.model.galaxies.left_lens.bulge.elliptical_comps
    )
    left_sersic.intensity = phase1.result.model.galaxies.left_lens.bulge.intensity
    left_sersic.effective_radius = (
        phase1.result.model.galaxies.left_lens.bulge.effective_radius
    )
    left_sersic.sersic_index = phase1.result.model.galaxies.left_lens.bulge.sersic_index

    left_mass = af.PriorModel(al.mp.EllipticalIsothermal)
    left_mass.elliptical_comps = (
        phase3.result.model.galaxies.left_lens.mass.elliptical_comps
    )
    left_mass.einstein_radius = (
        phase3.result.model.galaxies.left_lens.mass.einstein_radius
    )

    left_lens = al.GalaxyModel(
        redshift=redshift_lens, bulge=left_sersic, mass=left_mass
    )

    right_sersic = af.PriorModel(al.lp.EllipticalSersic)
    right_sersic.elliptical_comps = (
        phase2.result.model.galaxies.right_lens.bulge.elliptical_comps
    )
    right_sersic.intensity = phase2.result.model.galaxies.right_lens.bulge.intensity
    right_sersic.effective_radius = (
        phase2.result.model.galaxies.right_lens.bulge.effective_radius
    )
    right_sersic.sersic_index = (
        phase2.result.model.galaxies.right_lens.bulge.sersic_index
    )

    right_mass = af.PriorModel(al.mp.EllipticalIsothermal)
    right_mass.elliptical_comps = (
        phase3.result.model.galaxies.right_lens.mass.elliptical_comps
    )
    right_mass.einstein_radius = (
        phase3.result.model.galaxies.right_lens.mass.einstein_radius
    )

    right_lens = al.GalaxyModel(
        redshift=redshift_lens, bulge=right_sersic, mass=right_mass
    )

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_light_x2[bulge]_mass_x2[sie]_source[exp]",
            n_live_points=60,
            evidence_tolerance=0.3,
        ),
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=phase3.result.model.galaxies.source,
        ),
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, phase1, phase2, phase3, phase4
    )
