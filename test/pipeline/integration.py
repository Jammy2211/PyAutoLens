from autolens.pipeline import pipeline as pl
from autolens.pipeline import phase as ph
from autolens.imaging import image as im
from autolens.imaging import scaled_array
from autolens.analysis import galaxy_prior as gp
from autolens.profiles import mass_profiles
from autolens.autopipe import model_mapper as mm
from autolens.autopipe import non_linear as nl
import shutil
import numpy as np

import os

dirpath = os.path.dirname(os.path.realpath(__file__))


def load_image(name):
    # Load up the weighted_data
    data_dir = "{}/../../data/{}".format(dirpath, name)

    data = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/image', hdu=0, pixel_scale=0.1)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0, pixel_scale=0.1)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0, pixel_scale=0.1)

    return im.Image(array=data, pixel_scale=0.05, psf=psf, noise=noise)


def test_source_only_phase_1():
    phase1 = pl.make_source_only_pipeline().phases[0]

    result = phase1.run(load_image('source_sersic'))
    print(result)


def test_source_only_phase_2():
    lens_galaxy = gp.GalaxyPrior(
        sie=mass_profiles.SphericalIsothermal,
        shear=mass_profiles.ExternalShear)

    variable = mm.ModelMapper()
    variable.lens_galaxy = lens_galaxy

    last_result = ph.SourceLensPhase.Result(mm.ModelInstance(), 1., variable,
                                            galaxy_images=[np.ones(1264), np.ones(1264)])

    phase2 = pl.make_source_only_pipeline().phases[1]

    result = phase2.run(mass_profiles, last_result)
    print(result)


def test_source_only_pipeline():
    pipeline = pl.make_source_only_pipeline()
    results = pipeline.run(load_image('source_sersic'))
    for result in results:
        print(result)


def test_profile_pipeline():
    name = "test_pipeline"
    try:
        shutil.rmtree("{}/../../output/{}".format(dirpath, name))
    except FileNotFoundError:
        pass
    pipeline = pl.make_profile_pipeline(name, optimizer_class=nl.DownhillSimplex)
    results = pipeline.run(load_image("integration/hst_0"))
    for result in results:
        print(result)


if __name__ == "__main__":
    test_profile_pipeline()
