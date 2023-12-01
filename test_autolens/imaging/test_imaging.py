import autolens as al
import numpy as np


def test__via_tracer_from():
    psf = al.Kernel2D.from_gaussian(
        shape_native=(7, 7), sigma=0.5, pixel_scales=1.0
    )

    grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(intensity=1.0),
        mass=al.mp.Isothermal(einstein_radius=1.6),
    )

    source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.Sersic(intensity=0.3))

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise=False,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset_via_image = simulator.via_image_from(
        image=tracer.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (20, 20)
    assert dataset.data.native[0, 0] != dataset_via_image.data.native[0, 0]
    assert dataset.data.native[10, 10] == dataset_via_image.data.native[10, 10]
    assert (dataset.psf == dataset_via_image.psf).all()
    assert (dataset.noise_map == dataset_via_image.noise_map).all()

def test__via_deflections_and_galaxies_from():

    psf = al.Kernel2D.no_blur(pixel_scales=0.05)

    grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5, mass=al.mp.Isothermal(einstein_radius=1.6)
    )

    source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.Sersic(intensity=0.3))

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise=False,
    )

    dataset = simulator.via_deflections_and_galaxies_from(
        deflections=tracer.deflections_yx_2d_from(grid=grid),
        galaxies=[source_galaxy],
    )

    dataset_via_image = simulator.via_image_from(
        image=tracer.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (20, 20)
    assert (dataset.data.native == dataset_via_image.data.native).all()
    assert (dataset.psf == dataset_via_image.psf).all()
    assert (dataset.noise_map == dataset_via_image.noise_map).all()

def test__via_galaxies():

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(
            centre=(0.1, 0.1),
            ell_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = al.Kernel2D.no_mask(values=[[1.0]], pixel_scales=0.2)

    simulator = al.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise=True,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(
        galaxies=[lens_galaxy, source_galaxy], grid=grid
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    dataset_via_image = simulator.via_image_from(
        image=tracer.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (11, 11)
    assert (dataset.data.native == dataset_via_image.data.native).all()
    assert (dataset.psf == dataset_via_image.psf).all()
    assert (dataset.noise_map == dataset_via_image.noise_map).all()

def test__via_source_image_from():
    
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(
            intensity=0.3,
        ),
    )
    
    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = al.Kernel2D.no_mask(values=[[1.0]], pixel_scales=0.2)

    simulator = al.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise=True,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(
        galaxies=[lens_galaxy, source_galaxy], grid=grid
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    traced_grid = tracer.traced_grid_2d_list_from(grid=grid)[-1]

    source_image = source_galaxy.image_2d_from(grid=traced_grid)

    dataset_via_image = simulator.via_source_image_from(
        tracer=tracer,
        grid=grid,
        source_image=source_image        
    )

    assert dataset.shape_native == (11, 11)
    assert (dataset.data.native == dataset_via_image.data.native).all()
    assert (dataset.psf == dataset_via_image.psf).all()
    assert (dataset.noise_map == dataset_via_image.noise_map).all()