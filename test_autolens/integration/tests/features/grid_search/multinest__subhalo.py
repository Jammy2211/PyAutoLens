import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest_grid__subhalo"
data_type = "lens_sie__source_smooth"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):

    lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)

    lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    lens.mass.axis_ratio = af.UniformPrior(lower_limit=0.65, upper_limit=0.75)
    lens.mass.phi = af.UniformPrior(lower_limit=40.0, upper_limit=50.0)
    lens.mass.einstein_radius = af.UniformPrior(lower_limit=1.55, upper_limit=1.65)

    source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

    source.light.centre_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    source.light.centre_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    source.light.axis_ratio = af.UniformPrior(lower_limit=0.75, upper_limit=0.85)
    source.light.phi = af.UniformPrior(lower_limit=50.0, upper_limit=70.0)
    source.light.intensity = af.UniformPrior(lower_limit=0.35, upper_limit=0.45)
    source.light.effective_radius = af.UniformPrior(lower_limit=0.45, upper_limit=0.55)
    source.light.sersic_index = af.UniformPrior(lower_limit=0.9, upper_limit=1.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens, source=source),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=False)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(
        redshift=0.5, mass=al.mp.SphericalTruncatedNFWMassToConcentration
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    phase2 = GridPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens,
            subhalo=subhalo,
            source=af.last.instance.galaxies.source,
        ),
        optimizer_class=optimizer_class,
        number_of_steps=2,
    )

    phase2.optimizer.const_efficiency_mode = True

    phase3 = al.PhaseImaging(
        phase_name="phase_3__subhalo_refine",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last[-1].model.galaxies.lens,
            subhalo=phase2.result.model.galaxies.subhalo,
            source=af.last[-1].instance.galaxies.source,
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
