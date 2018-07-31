from src.pipeline import pipeline as pl
from src.pipeline import phase as ph
from src.imaging import image as im
from src.imaging import scaled_array
from src.analysis import galaxy_prior as gp
from src.profiles import mass_profiles
from src.autopipe import model_mapper as mm
import numpy as np

import os

dirpath = os.path.dirname(os.path.realpath(__file__))


def make_toy_image():
    # Load up the weighted_data
    lens_name = 'source_sersic'
    data_dir = "{}/../../data/{}".format(dirpath, lens_name.format(os.path.dirname(os.path.realpath(__file__))))

    data = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/image', hdu=0, pixel_scale=0.1)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0, pixel_scale=0.1)
    exposure_time = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/exposure_time', hdu=0,
                                                       pixel_scale=0.1)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0, pixel_scale=0.1)

    return im.Image(array=data, effective_exposure_time=exposure_time, pixel_scale=0.1, psf=psf,
                    background_noise=noise, poisson_noise=noise)


def test_source_only_phase_1():
    phase1 = pl.make_source_only_pipeline().phases[0]

    result = phase1.run(make_toy_image())
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
    results = pipeline.run(make_toy_image())
    for result in results:
        print(result)


def test_profile_pipeline():
    pipeline = pl.make_profile_pipeline()
    results = pipeline.run(image)
    for result in results:
        print(result)


if __name__ == "__main__":
    test_source_only_pipeline()
