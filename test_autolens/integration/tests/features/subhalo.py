import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest_grid__subhalo"
dataset_name = "lens_sie__source_smooth"
instrument = "vro"


def make_pipeline(name, path_prefix, search=af.DynestyStatic()):

    lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)

    lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    lens.mass.einstein_radius = af.UniformPrior(lower_limit=1.55, upper_limit=1.65)

    source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

    source.light.centre_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    source.light.centre_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    source.light.intensity = af.UniformPrior(lower_limit=0.35, upper_limit=0.45)
    source.light.effective_radius = af.UniformPrior(lower_limit=0.45, upper_limit=0.55)
    source.light.sersic_index = af.UniformPrior(lower_limit=0.9, upper_limit=1.1)

    phase1 = al.PhaseImaging(
        name="phase[1]",
        path_prefix=path_prefix,
        galaxies=dict(lens=lens, source=source),
        search=search,
        settings=al.SettingsPhaseImaging(),
    )

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=False)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalTruncatedNFWMCRLudlow)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)

    phase2 = GridPhase(
        name="phase[2]",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens,
            subhalo=subhalo,
            source=af.last.instance.galaxies.source,
        ),
        search=search,
        settings=al.SettingsPhaseImaging(),
        number_of_steps=2,
    )

    phase3 = al.PhaseImaging(
        name="phase_3__subhalo_refine",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=af.last[-1].model.galaxies.lens,
            subhalo=phase2.result.model.galaxies.subhalo,
            source=af.last[-1].instance.galaxies.source,
        ),
        settings=al.SettingsPhaseImaging(),
        search=af.DynestyStatic(),
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
