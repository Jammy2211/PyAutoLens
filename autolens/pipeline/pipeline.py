import autofit as af
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg


class PipelineHyperSettings(object):
    def __init__(self, galaxies=False, image_sky=False, background_noise=False):

        self.galaxies = galaxies
        self.image_sky = image_sky
        self.background_noise = background_noise


class PipelineSourceSettings(object):
    def __init__(
        self,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        align_light_mass_centre=True,
        fix_lens_light=True,
    ):

        self.pixelization = pixelization
        self.regularization = regularization
        self.align_light_mass_centre = align_light_mass_centre
        self.fix_lens_light = fix_lens_light


class PipelineLightSettings(object):
    def __init__(
        self,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
    ):

        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_phi = align_bulge_disk_phi
        self.align_bulge_disk_axis_ratio = align_bulge_disk_axis_ratio
        self.disk_as_sersic = disk_as_sersic


class PipelineMassSettings(object):
    def __init__(
        self,
        include_shear=False,
        fix_lens_light=False,
        align_light_dark_centre=True,
        align_bulge_dark_centre=True,
    ):

        self.include_shear = include_shear
        self.fix_lens_light = fix_lens_light
        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre


class PipelineDataset(af.Pipeline):
    def __init__(self, pipeline_name, pipeline_tag, *phases, hyper_mode=False):

        super(PipelineDataset, self).__init__(pipeline_name, pipeline_tag, *phases)

        self.hyper_mode = hyper_mode

    def run(self, dataset, mask=None, positions=None, data_name=None):

        if self.hyper_mode and mask is None:
            raise exc.PhaseException(
                "The pipeline is running in hyper_galaxies mode, but has not received an input mask. Add"
                "a mask to the run function of the pipeline (e.g. pipeline.run(data_type=data_type, mask=mask)"
            )

        def runner(phase, results):
            return phase.run(
                dataset=dataset, results=results, mask=mask, positions=positions
            )

        return self.run_function(runner, data_name)


class PipelinePositions(af.Pipeline):
    def run(self, positions, pixel_scales):
        def runner(phase, results):
            return phase.run(
                positions=positions, pixel_scales=pixel_scales, results=results
            )

        return self.run_function(runner)
