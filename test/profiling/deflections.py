from astropy.io import fits
import os
import numpy as np
import time

from autolens.data.array import mask as ma
from autolens.data.imaging import image as im
from autolens.data.imaging.plotters import imaging_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.data.array import grids
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.model.inversion.util import inversion_util
from autolens.model.inversion.util import regularization_util
from autolens.lensing import lensing_fitting

image_shape = (50, 50)
pixel_scale = 0.02
psf_shape = (21, 21)

grids = grids.DataGrids.grids_for_simulation(shape=image_shape, sub_grid_size=2,
                                             pixel_scale=pixel_scale, psf_shape=psf_shape)

print('Number of points = ', grids.sub.shape[0])

sie_mass_profile = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.8, phi=45.0)

start = time.time()
sie_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("SIE time = {}".format(diff))

sple_mass_profile = mp.EllipticalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.8, phi=45.0, slope=1.5)

start = time.time()
sple_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("SPLE time = {}".format(diff))

sple_core_mass_profile = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.8, phi=45.0,
                                                    slope=1.5, core_radius=0.01)

start = time.time()
sple_core_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("SPLE Core time = {}".format(diff))

nfw_sph_mass_profile = mp.SphericalNFW(kappa_s=0.1, scale_radius=10.0)

start = time.time()
nfw_sph_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("NEW Sph time = {}".format(diff))

nfw_ell_mass_profile = mp.EllipticalNFW(kappa_s=0.1, scale_radius=10.0, axis_ratio=0.7, phi=45.0)

start = time.time()
nfw_ell_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("NEW Ell time = {}".format(diff))

dev_mass_profile = mp.EllipticalDevVaucouleurs(intensity=1.0, effective_radius=1.0, axis_ratio=0.8, phi=45.0)

start = time.time()
dev_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("Dev time = {}".format(diff))

sersic_mass_profile = mp.EllipticalSersic(intensity=1.0, effective_radius=1.0, sersic_index=2.0, axis_ratio=0.8,
                                          phi=45.0)

start = time.time()
sersic_mass_profile.deflections_from_grid(grid=grids.sub)
diff = time.time() - start
print("Sersic time = {}".format(diff))

def func1(x, y, z, s, d):
    return x+1

x=0
start = time.time()
for i in range(100*grids.sub.shape[0]):
    x = func1(x, x, x, x, x)
diff = time.time() - start
print("Loop time = {}".format(diff))
