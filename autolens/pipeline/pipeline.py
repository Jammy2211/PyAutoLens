from autofit.tools.pipeline import Pipeline


class PipelineImaging(Pipeline):
    def run(self, data, mask=None, positions=None, data_name=None):
        def runner(phase, results):
            return phase.run(data=data, results=results, mask=mask, positions=positions)

        return self.run_function(runner, data_name)


class PipelinePositions(Pipeline):
    def run(self, positions, pixel_scale):
        def runner(phase, results):
            return phase.run(positions=positions, pixel_scale=pixel_scale, results=results)

        return self.run_function(runner)

def bin_up_factor_tag_from_bin_up_factor(bin_up_factor):
    """Generate a bin up tag, to customize phase names based on the bin up factor used in a pipeline. This changes
    the phase name 'phase_name' as follows:

    bin_up_factor = 1 -> phase_name
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    """
    if bin_up_factor == 1:
        return ''
    else:
        return '_bin_up_factor_' + str(bin_up_factor)


def interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale):
    """Generate a bin up tag, to customize phase names based on the bin up factor used in a pipeline. This changes
    the phase name 'phase_name' as follows:

    interp_pixel_scale = 1 -> phase_name
    interp_pixel_scale = 2 -> phase_name_interp_pixel_scale_2
    interp_pixel_scale = 2 -> phase_name_interp_pixel_scale_2
    """
    if interp_pixel_scale is None:
        return ''
    else:
        return '_interp_pixel_scale_' + str(interp_pixel_scale)