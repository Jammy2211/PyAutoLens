import autofit as af
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg


class PipelineSettings(object):
    def __init__(
        self,
        include_shear=False,
        fix_lens_light=False,
        pixelization=pix.VoronoiMagnification,
        regularization=reg.Constant,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
        align_light_dark_centre=True,
        align_bulge_dark_centre=True,
    ):

        self.include_shear = include_shear
        self.fix_lens_light = fix_lens_light
        self.pixelization = pixelization
        self.regularization = regularization
        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_phi = align_bulge_disk_phi
        self.align_bulge_disk_axis_ratio = align_bulge_disk_axis_ratio
        self.disk_as_sersic = disk_as_sersic
        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre


class PipelineSettingsHyper(PipelineSettings):
    def __init__(
        self,
        hyper_galaxies=True,
        hyper_image_sky=False,
        hyper_background_noise=False,
        include_shear=False,
        fix_lens_light=False,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
        align_light_dark_centre=True,
        align_bulge_dark_centre=True,
    ):

        super(PipelineSettingsHyper, self).__init__(
            include_shear=include_shear,
            fix_lens_light=fix_lens_light,
            pixelization=pixelization,
            regularization=regularization,
            align_bulge_disk_centre=align_bulge_disk_centre,
            align_bulge_disk_phi=align_bulge_disk_phi,
            align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
            disk_as_sersic=disk_as_sersic,
            align_light_dark_centre=align_light_dark_centre,
            align_bulge_dark_centre=align_bulge_dark_centre,
        )

        self.hyper_galaxies = hyper_galaxies
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise


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
