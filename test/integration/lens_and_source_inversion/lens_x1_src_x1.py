from autolens.pipeline import pipeline as pl
from autolens.pipeline import phase as ph
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.inversion import pixelizations as pix
from autolens.lensing import galaxy_model as gm
from autolens.autofit import non_linear as nl
from autolens.autofit import model_mapper as mm
from autolens.lensing import galaxy
from autolens import conf
from test.integration import tools

import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '/gpfs/data/pdtw24/Lens/int/lens_and_source_inversion/'

def test_lens_x1_src_x1_pix_pipeline():

    pipeline_name = "l1_s1"
    data_name = '/l1_s1'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    lens_mass = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light = lp.EllipticalSersic(centre=(-0.01, -0.01), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                       effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sie=lens_mass)
    source_galaxy = galaxy.Galaxy(sersic=source_light)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy], target_signal_to_noise=30.0)

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path + pipeline_name)
    except FileNotFoundError:
        pass

    pipeline = make_lens_x1_src_x1_pix_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_x1_src_x1_pix_pipeline(pipeline_name):

    class SourcePix(ph.LensMassAndSourcePixelizationPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sie.centre.centre_0 = mm.UniformPrior(0.0, 0.02)
            self.lens_galaxies[0].sie.centre.centre_1 = mm.UniformPrior(0.0, 0.02)
            self.lens_galaxies[0].sie.einstein_radius = mm.Constant(1.6)
            self.source_galaxies[0].mapper.shape.shape_0 = mm.UniformPrior(19.5, 20.5)
            self.source_galaxies[0].mapper.shape.shape_1 = mm.UniformPrior(19.5, 20.5)

    phase1 = SourcePix(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                       source_galaxies=[gm.GalaxyModel(pixelization=pix.RectangularRegConst)],
                       optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1)


if __name__ == "__main__":
    test_lens_x1_src_x1_pix_pipeline()
