import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using an _EllipticalIsothermal_ _MassProfile_ and a source composed of 4 
parametric _EllipticalSersic_'s.

The pipeline is four phases:

Phase 1:

    Fit the _EllipticalIsothermal_ mass model and the first _EllipticalSersic_ light profile of the source.

    Lens Light: None
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Add the second _EllipticalSersic_ to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: _EllipticalSersic_ + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 1), Source Light (model -> phase 1).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

Phase 3:

    Add the third _EllipticalSersic_ to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: _EllipticalSersic_ + _EllipticalSersic_ + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 2), Source Light (model -> phase 2).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

Phase 4:

    Add the fourth _EllipticalSersic_ to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: _EllipticalSersic_ + _EllipticalSersic_ + _EllipticalSersic_ + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 3), Source Light (model -> phase 3).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__complex_source"

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1: Initialize the lens's mass model using a simple source.
    
    This won't fit the complicated structure of the source, but it'll give us a reasonable estimate of the
    einstein radius and the other lens-mass parameters.
    """

    phase1 = al.PhaseImaging(
        phase_name="phase_1__mass_sie__source_x1_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, sersic_0=al.lp.EllipticalSersic),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=5.0),
    )

    """
    Phase 1: Add a second source component, using the previous model as the initialization on the lens / source
             parameters. We'll vary the parameters of the lens mass model and first source galaxy component during the 
             fit.
    """

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_sie__source_sersic_x2",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                sersic_0=phase1.result.model.galaxies.source.sersic_0,
                sersic_1=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=5.0),
    )

    """Phase 3: Same again, but with 3 source galaxy components."""

    phase3 = al.PhaseImaging(
        phase_name="phase_3__mass_sie__source_sersic_x3",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase2.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                sersic_0=phase2.result.model.galaxies.source.sersic_0,
                sersic_1=phase2.result.model.galaxies.source.sersic_1,
                sersic_2=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, evidence_tolerance=5.0),
    )

    """Phase 4: And one more for luck!"""

    phase4 = al.PhaseImaging(
        phase_name="phase_4__mass_sie__source_sersic_x4",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase3.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                sersic_0=phase3.result.model.galaxies.source.sersic_0,
                sersic_1=phase3.result.model.galaxies.source.sersic_1,
                sersic_2=phase3.result.model.galaxies.source.sersic_2,
                sersic_3=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, evidence_tolerance=0.3),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
