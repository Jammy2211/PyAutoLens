import os

import shutil

import autofit as af
import autolens as al
from autolens.array.util import array_util
from autolens.model.galaxy import galaxy
from test.integration import integration_util

dirpath = os.path.dirname(os.path.realpath(__file__))
af.conf.instance = af.conf.Config(
    "{}/config".format(dirpath), "{}/output/".format(dirpath)
)

dirpath = os.path.dirname(dirpath)
output_path = "{}/output".format(dirpath)

test_name = "tests"


def simulate_integration_image(test_name, pixel_scales, galaxies):
    output_path = (
        "{}/test_files/simulator/".format(os.path.dirname(os.path.realpath(__file__)))
        + test_name
        + "/"
    )
    psf_shape_2d = (11, 11)
    image_shape = (150, 150)

    psf = al.kernel.from_gaussian(
        shape_2d=psf_shape_2d, pixel_scales=pixel_scales, sigma=pixel_scales
    )

    grid = al.Grid.from_shape_2d_pixel_scale_and_sub_size(
        shape_2d=image_shape, pixel_scales=pixel_scales, sub_size=1
    )

    tracer = al.Tracer.from_galaxies(galaxies=galaxies)

    ### Setup as a simulated image_coords and output as a fits for an lens ###

    imaging_simulated = al.SimulatedImagingData.from_tracer(
        tracer=tracer,
        pixel_scales=pixel_scales,
        exposure_time=100.0,
        background_level=10.0,
        psf=psf,
        noise_seed=1,
    )

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    array_util.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.image,
        file_path=output_path + "/image.fits",
        overwrite=True,
    )
    array_util.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.noise_map,
        file_path=output_path + "/noise_map.fits",
        overwrite=True,
    )
    array_util.numpy_array_2d_to_fits(
        array_2d=psf, file_path=output_path + "/psf.fits", overwrite=True
    )
    array_util.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.exposure_time_map,
        file_path=output_path + "/exposure_map.fits",
        overwrite=True,
    )


class TestAdvancedModelMapper:
    def test_fully_qualified_paramnames(self):
        mapper = af.ModelMapper()
        galaxy_model = al.GalaxyModel(
            redshift=0.5, light_profile=al.lp.EllipticalLightProfile
        )
        light_profile = galaxy_model.light_profile
        mapper.galaxy_model = galaxy_model

        assert light_profile.name_for_prior(light_profile.axis_ratio) == "axis_ratio"
        assert light_profile.name_for_prior(light_profile.centre.centre_0) == "centre_0"

        assert (
            galaxy_model.name_for_prior(light_profile.axis_ratio)
            == "light_profile_axis_ratio"
        )

        assert mapper.param_names[0] == "galaxy_model_light_profile_centre_0"


class TestPhaseModelMapper:
    def test_pairing_works(self):

        test_name = "pair_floats"

        integration_util.reset_paths(test_name, output_path)

        sersic = al.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.8,
            phi=90.0,
            intensity=1.0,
            effective_radius=1.3,
            sersic_index=3.0,
        )

        lens_galaxy = al.Galaxy(redshift=0.5, light_profile=sersic)

        simulate_integration_image(
            test_name=test_name, pixel_scales=0.5, galaxies=[lens_galaxy]
        )

        path = "{}/".format(
            os.path.dirname(os.path.realpath(__file__))
        )  # Setup path so we can output the simulated image.

        imaging = al.imaging.from_fits(
            image_path=path + "/test_files/simulator/" + test_name + "/image.fits",
            psf_path=path + "/test_files/simulator/" + test_name + "/psf.fits",
            noise_map_path=path
            + "/test_files/simulator/"
            + test_name
            + "/noise_map.fits",
            real_space_pixel_scales=0.1,
        )

        class MMPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.sersic.intensity = (
                    self.galaxies.lens.sersic.axis_ratio
                )

        phase = MMPhase(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, sersic=al.lp.EllipticalSersic)
            ),
            optimizer_class=af.MultiNest,
            phase_name="{}/phase1".format(test_name),
        )

        initial_total_priors = phase.model.prior_count
        phase.make_analysis(dataset=imaging)

        assert phase.galaxies[0].sersic.intensity == al.Galaxies[0].sersic.axis_ratio
        assert initial_total_priors - 1 == phase.model.prior_count

        lines = list(
            filter(
                lambda line: "axis_ratio" in line or "intensity" in line,
                str(phase.model).split("\n"),
            )
        )

        assert len(lines) == 2
        assert (
            "galaxies_lens_sersic_axis_ratio                                                  UniformPrior, lower_limit = 0.2, upper_limit = 1.0"
            in lines
        )
        assert (
            "galaxies_lens_sersic_intensity                                                   UniformPrior, lower_limit = 0.2, upper_limit = 1.0"
            in lines
        )

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    def test_instances_work(self):
        name = "const_float"
        test_name = "/const_float"

        integration_util.reset_paths(test_name, output_path)

        sersic = al.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.8,
            phi=90.0,
            intensity=1.0,
            effective_radius=1.3,
            sersic_index=3.0,
        )

        lens_galaxy = al.Galaxy(redshift=0.5, light_profile=sersic)

        simulate_integration_image(
            test_name=test_name, pixel_scales=0.5, galaxies=[lens_galaxy]
        )
        path = "{}/".format(
            os.path.dirname(os.path.realpath(__file__))
        )  # Setup path so we can output the simulated image.

        imaging = al.imaging.from_fits(
            image_path=path + "/test_files/simulator/" + test_name + "/image.fits",
            psf_path=path + "/test_files/simulator/" + test_name + "/psf.fits",
            noise_map_path=path
            + "/test_files/simulator/"
            + test_name
            + "/noise_map.fits",
            real_space_pixel_scales=0.1,
        )

        class MMPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.sersic.axis_ratio = 0.2
                self.galaxies.lens.sersic.phi = 90.0
                self.galaxies.lens.sersic.intensity = 1.0
                self.galaxies.lens.sersic.effective_radius = 1.3
                self.galaxies.lens.sersic.sersic_index = 3.0

        phase = MMPhase(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, sersic=al.lp.EllipticalSersic)
            ),
            optimizer_class=af.MultiNest,
            phase_name="{}/phase1".format(name),
        )

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        phase.make_analysis(dataset=imaging)

        sersic = phase.model.galaxies[0].sersic

        assert isinstance(sersic, af.PriorModel)

        assert isinstance(sersic.axis_ratio, float)
        assert isinstance(sersic.phi, float)
        assert isinstance(sersic.intensity, float)
        assert isinstance(sersic.effective_radius, float)
        assert isinstance(sersic.sersic_index, float)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
