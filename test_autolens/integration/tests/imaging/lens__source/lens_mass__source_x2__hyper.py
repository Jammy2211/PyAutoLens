import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source"
test_name = "lens_mass__source_x2__hyper"
data_name = "lens_sie__source_smooth"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source_0=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=search,
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source_0=al.GalaxyModel(
                redshift=1.0, light=phase1.result.model.galaxies.source_0.light
            ),
            source_1=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=search,
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(hyper_galaxies_search=True)

    class HyperLensSourcePlanePhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.source_0.hyper_galaxy = (
                results.last.hyper_combined.instance.galaxies.source_0.hyper_galaxy
            )

            self.galaxies.source_1.hyper_galaxy = (
                results.last.hyper_combined.instance.galaxies.source_1.hyper_galaxy
            )

    phase3 = HyperLensSourcePlanePhase(
        phase_name="phase_3",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase2.result.model.galaxies.lens.mass
            ),
            source_0=al.GalaxyModel(
                redshift=1.0,
                light=phase2.result.model.galaxies.source_0.light,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.source_0.hyper_galaxy,
            ),
            source_1=al.GalaxyModel(
                redshift=1.0,
                light=phase2.result.model.galaxies.source_1.light,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.source_1.hyper_galaxy,
            ),
        ),
        search=search,
    )

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
