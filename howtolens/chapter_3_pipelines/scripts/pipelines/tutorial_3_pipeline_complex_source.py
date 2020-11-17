import autofit as af
from os import path
import autolens as al

"""
In this pipeline, we fit the a strong lens using an `EllipticalIsothermal` `MassProfile`.and a source composed of 4 
parametric `EllipticalSersic``..

The pipeline is four phases:

Phase 1:

    Fit the `EllipticalIsothermal` mass model and the first `EllipticalSersic` light profile of the source.

    Lens Light: None
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Add the second `EllipticalSersic` to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: `EllipticalSersic` + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 1), Source Light (model -> phase 1).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

Phase 3:

    Add the third `EllipticalSersic` to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: `EllipticalSersic` + `EllipticalSersic` + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 2), Source Light (model -> phase 2).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

Phase 4:

    Add the fourth `EllipticalSersic` to the source model.

    Lens Light: None
    Lens Mass: EllipticalIsothermal
    Source Light: `EllipticalSersic` + `EllipticalSersic` + `EllipticalSersic` + EllipticalSersic
    Prior Passing: Lens Mass (model -> phase 3), Source Light (model -> phase 3).
    Notes: Uses the previous mass model and source model to initialize the non-linear search.

"""


def make_pipeline(path_prefix, settings, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__complex_source"

    """
    Phase 1: Initialize the lens`s mass model using a simple source.
    
    This won't fit the complicated structure of the source, but it`ll give us a reasonable estimate of the
    einstein radius and the other lens-mass parameters.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]__mass[sie]__source_x1[bulge]",
            n_live_points=40,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=al.mp.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, bulge_0=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """
    Phase 1: Add a second source component, using the previous model as the initialization on the lens / source
             parameters. we'll vary the parameters of the lens mass model and first source galaxy component during the 
             fit.
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_mass[sie]_source_x2[bulge]",
            n_live_points=40,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                bulge_0=phase1.result.model.galaxies.source.bulge_0,
                bulge_1=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
    )

    """Phase 3: Same again, but with 3 source galaxy components."""

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_mass[sie]_source_x3[bulge]",
            n_live_points=50,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=phase2.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                bulge_0=phase2.result.model.galaxies.source.bulge_0,
                bulge_1=phase2.result.model.galaxies.source.bulge_1,
                bulge_2=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
    )

    """Phase 4: And one more for luck!"""

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_mass[sie]_source_x4[bulge]",
            n_live_points=50,
            evidence_tolerance=0.3,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=phase3.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                bulge_0=phase3.result.model.galaxies.source.bulge_0,
                bulge_1=phase3.result.model.galaxies.source.bulge_1,
                bulge_2=phase3.result.model.galaxies.source.bulge_2,
                bulge_3=al.lp.EllipticalSersic,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, phase1, phase2, phase3, phase4
    )
