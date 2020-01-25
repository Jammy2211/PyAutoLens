import autofit as af
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg


class PipelineGeneralSettings(object):
    def __init__(self, hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False, with_shear=True):

        self.hyper_galaxies = hyper_galaxies
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise
        self.with_shear = with_shear


class PipelineSourceSettings(object):
    def __init__(
        self,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        lens_light_centre=None,
        lens_mass_centre=None,
        align_light_mass_centre=True,
        fix_lens_light=True,
    ):

        self.pixelization = pixelization
        self.regularization = regularization
        self.lens_light_centre = lens_light_centre
        self.lens_mass_centre = lens_mass_centre
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
        fix_lens_light=False,
        align_light_dark_centre=True,
        align_bulge_dark_centre=True,
    ):

        self.fix_lens_light = fix_lens_light
        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre


class PipelineDataset(af.Pipeline):
    def __init__(self, pipeline_name, pipeline_tag, *phases):

        super(PipelineDataset, self).__init__(pipeline_name, pipeline_tag, *phases)

    def run(self, dataset, mask, positions=None, data_name=None):

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
