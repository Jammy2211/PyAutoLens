import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data import ccd
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from test.integration import tools

test_type = 'sensitivity'
test_name = "sensitivity_profile_via_multinest"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    lens_mass = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)
    lens_subhalo = mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=0.0)
    source_light = lp.SphericalSersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(mass=lens_mass, subhalo=lens_subhalo)
    source_galaxy = galaxy.Galaxy(light=source_light)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy], target_signal_to_noise=30.0)

    ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + test_name + '/image.fits',
                                        psf_path=path + '/data/' + test_name + '/psf.fits',
                                        noise_map_path=path + '/data/' + test_name + '/noise_map.fits',
                                        pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class SensitivePhase(ph.SensitivityPhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = 0.0
            self.lens_galaxies.lens.mass.centre_1 = 0.0
            self.lens_galaxies.lens.mass.einstein_radius = 1.6

            self.source_galaxies.source.light.centre_0 = 0.0
            self.source_galaxies.source.light.centre_1 = 0.0
            self.source_galaxies.source.light.intensity = 1.0
            self.source_galaxies.source.light.effective_radius = 0.5
            self.source_galaxies.source.light.sersic_index = 1.0

            self.sensitive_galaxies.subhalo.mass.centre_0 = prior.GaussianPrior(mean=0.0, sigma=1.0)
            self.sensitive_galaxies.subhalo.mass.centre_1 = prior.GaussianPrior(mean=0.0, sigma=1.0)
            self.sensitive_galaxies.subhalo.mass.kappa_s = 0.1
            self.sensitive_galaxies.subhalo.mass.scale_radius = 5.0

    phase1 = SensitivePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.SphericalIsothermal)),
                            source_galaxies=dict(source=gm.GalaxyModel(light=lp.SphericalSersic)),
                            sensitive_galaxies=dict(subhalo=gm.GalaxyModel(mass=mp.SphericalNFW)),
                            optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
