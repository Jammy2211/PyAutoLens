# from src.pipeline import phase as ph
# from src.analysis import galaxy_prior as gp


class Pipeline(object):
    def __init__(self, *phases):
        self.phases = phases
        self.results = []

    @property
    def last_result(self):
        return None if len(self.results) == 0 else self.results[-1]

    def run(self, image):
        for phase in self.phases:
            self.results.append(phase.run(image, self.last_result))

# class ExtendedPhase(ph.SourceLensPhase):
#     def pass_priors(self, last_results):
#         self.lens_galaxy = last_results.constant.lens_galaxy
#
#     def customize_image(self, masked_image, last_result):
#         return masked_image
#
#
# img = None
#
# first_phase = ph.SourceLensPhase(lens_galaxy=gp.GalaxyPrior(), source_galaxy=gp.GalaxyPrior())
# second_phase = ExtendedPhase(source_galaxy=gp.GalaxyPrior())
#
#
# pipeline = Pipeline(first_phase, second_phase)
#
# pipeline.run(img)
