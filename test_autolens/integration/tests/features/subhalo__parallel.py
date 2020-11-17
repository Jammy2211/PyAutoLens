import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest_grid_subhalo__parallel"
dataset_name = "mass_sie__source_sersic"
instrument = "vro"


def make_pipeline(name, path_prefix):

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

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=True)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalTruncatedNFWMCRLudlow)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)
    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)

    phase1 = GridPhase(
        name="phase[1]",
        galaxies=dict(lens=lens, subhalo=subhalo, source=source),
        search=search,
        settings=al.SettingsPhaseImaging(),
        number_of_steps=2,
    )

    return al.PipelineDataset(name, path_prefix, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
