import autofit as af
from os import path
import autolens as al

"""
In this pipeline, we fit the a strong lens using a `EllipticalIsothermal` `MassProfile`.and a source which uses an
inversion.

The pipeline is three phases:

Phase 1:

    Fit the lens mass model and source `LightProfile`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source `Inversion` using the lens `MassProfile` inferred in phase 1.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens & Mass (instance -> phase1).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 3:

    Refines the lens light and mass models using the source `Inversion` of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (model -> phase 1), Source `Inversion` (instance -> phase 2)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__inversion"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an  `ExternalShear`.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 3 & 4).
    """

    path_prefix = f"{setup.path_prefix}/{pipeline_name}/{setup.tag}"

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_mass[sie]_source[bulge]",
            n_live_points=50,
            evidence_tolerance=5.0,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic),
        ),
        settings=settings,
    )

    phase1.search.facc = 0.3
    phase1.search.const_efficiency_mode = True

    """
    Phase 2: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s `MassProfile`'s to the results of phase 1.
    """

    source = al.GalaxyModel(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
    )

    """We can customize the inversion`s priors like we do our light and mass profiles."""

    source.pixelization.shape_0 = af.UniformPrior(lower_limit=20.0, upper_limit=40.0)
    source.pixelization.shape_1 = af.UniformPrior(lower_limit=20.0, upper_limit=40.0)

    """
    The expected value of the regularization_coefficient depends on the details of the dataset reduction and
    source galaxy. A broad log-uniform prior is thus an appropriate way to sample the large range of
    possible values.
    """

    source.regularization.coefficient = af.LogUniformPrior(
        lower_limit=1.0e-6, upper_limit=10000.0
    )

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_source[inversion_initialize]", n_live_points=50
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=source,
        ),
        settings=settings,
    )

    """
    We now `extend` phase 1 with an additional `inversion phase` which uses the maximum log likelihood mass model of 
    phase 1 above to refine the `Inversion`, by fitting only the parameters of the `Pixelization` and _Regularization_
    (in this case, the shape of the `VoronoiMagnification` and `Regularization` coefficient of the `Constant`.

    The the `Inversion` phase results are accessible as attributes of the phase results and used in phase 3 below.
    """

    phase2 = phase2.extend_with_inversion_phase(
        hyper_search=af.DynestyStatic(n_live_points=50)
    )

    """
    Phase 3: Fit the lens`s mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of the extended `Inversion` phase of phase 2.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of phase 1.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_mass[sie]_source[inversion]", n_live_points=50
        ),
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.inversion.instance.galaxies.source.pixelization,
                regularization=phase2.result.inversion.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(pipeline_name, path_prefix, phase1, phase2, phase3)
