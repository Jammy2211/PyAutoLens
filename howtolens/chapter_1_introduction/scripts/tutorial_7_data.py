# %%
"""
Tutorial 7: Data
================

In this example, we'll use the `imaging` module to simulate `Imaging` of a strong lens made using a `Tracer`. By
simulate, we mean that it will appear as if we had observed it using a real telescope, with this example making an
image representative of Hubble Space Telescope imaging.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

# %%
"""
To simulate an image, we need to model the telescope`s optics. we'll do this by convolving the image with a 
Point-Spread Function using a `Kernel` object, which we can simulate as a Gaussian.
"""

# %%
psf = al.Kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

# %%
"""
To simulate `Imaging` dataset, we use a `Grid`, like usual.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1, sub_size=2)

# %%
"""
Now, lets setup our lens galaxy, source galaxy and `Tracer`.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
Lets look at the `Tracer`'s image - this is the image we'll be simulating.
"""

# %%
aplt.Tracer.image(tracer=tracer, grid=grid)

# %%
"""
To Simulate the `Imaging` data, we don't use the image plotted above. Instead, we use an image which has been generated
specifically for simulating an image, which pads the array it is computed on based on the shape of the PSF we
 convolve the image with. This ensures edge-effects do not degrade our simulation`s PSF convolution.
"""

# %%
normal_image = tracer.image_from_grid(grid=grid)
padded_image = tracer.padded_image_from_grid_and_psf_shape(
    grid=grid, psf_shape_2d=psf.shape_2d
)
print(normal_image.shape)
print(padded_image.shape)

# %%
"""
Now, to simulate the `Imaging` data, we pass the `Tracer` and `Grid` to the `Imaging` module`s simulate function. This adds
the following effects to the image:

 1) Telescope optics: Using the Point Spread Function above.
 2) The Background Sky: Although the image that is returned is automatically background sky subtracted.
 3) Poisson noise: Due to the background sky, lens galaxy and source galaxy Poisson photon counts.
"""

# %%
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

# %%
"""
Lets plot the image - we can see the image has been blurred due to the telescope optics and noise has been added.
"""

# %%
aplt.Imaging.image(imaging=imaging)

# %%
"""
Output these files to.fits files, we'll begin to analyze them in the next tutorial!
"""

# %%
"""
The `dataset_path` specifies where the data is output, this time in the directory `chapter_path/data`.

"""
dataset_path = path.join("dataset", "howtolens", "chapter_1")
print("Dataset Path: ", dataset_path)

# %%
"""
Now output our simulated data to hard-disk.
"""

# %%
imaging.output_to_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    overwrite=True,
)
