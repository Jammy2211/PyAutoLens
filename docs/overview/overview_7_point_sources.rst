.. _overview_7_point_sources:

Point Sources
-------------

So far, we have shown strongly lensed galaxies whose extended surface brightness is lensed into the awe-inspiring
giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where the background
source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, we do not want to model the source using a ``LightProfile`` which implicitly assumes an extended
surface brightness distribution. Instead, we assume that our source is a point source with a centre (y,x). Our
ray-tracing calculations no longer trace extended light rays from the source plane to the image-plane, but instead
now find the locations the point-source's multiple images appear in the image-plane.

Here is an example of a compact source that has been simulated in **PyAutoLens** and the positions of its four multiple
images:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/chi_squared_map_real.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/chi_squared_map_real.png
  :width: 400
  :alt: Alternative text

For point source modeling, our goal is to find the multiple images of a lens mass model given a (y,x) coordinate in the
source plane. This is an iterative problem performed in a very different way to ray-tracing used when evaluating a
``LightProfile``.

To treat a source as a point source, we create it as a galaxy using ``PointSource`` object and pass it to a tracer:

.. code-block:: bash

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllIsothermal(
            centre=(0.001, 0.001), einstein_radius=1.0, elliptical_comps=(0.0, 0.111111)
        )
    )

    point_source = al.ps.PointSource(centre=(0.07, 0.07))

    source_galaxy = al.Galaxy(redshift=1.0, point=point_source)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

This tracer allows us to compute deflection angles and therefore map image-pixels to the source-plane. We can therefore
pass it to the ``PositionSolver`` object to solve for the image-pixel coordinates which correspond to the source's
source-plane position at (0.07", 0.07").

There are many options that can be passed to a ``PositionsSovler``, below we only specify the precision we want the
image-plane coordinates to be calculated too, which is 0.001", much lower than the resolution fo the data we observed
the point-sources on!

.. code-block:: bash

    solver = al.PositionsSolver(
        grid=grid,
        pixel_scale_precision=0.001,
    )

    positions = solver.solve(lensing_obj=tracer, source_plane_coordinate=(0.07, 0.07))

Here is what the solved positions look like, compared to the observe data. In this example, the data was simulated
with the mass-model we used above, so the match is good:L

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/chi_squared_map_real.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** has full support for modeling strong lens datasets as a point-source. This might be used for analysing
strongly lensed quasars or supernovae, which are so compact we do not observe their extended emission.

To perform point-source modeling, we first create a ``PointSourceDataset`` containing the image-plane (y,x) positions
of each multiple image and their noise values (which would be the resolution of the imaging data they are observed).

The positions below correspond to those of an ``EllIsothermal`` mass model.

.. code-block:: bash

    point_source_dataset = al.PointSourceDataset(
        name="point_0",
        positions=al.Grid2DIrregular(
            [[1.1488, -1.1488], [1.109, 1.109], [-1.109, -1.109], [-1.1488, 1.1488]]
        ),
        positions_noise_map=al.ValuesIrregular([0.05, 0.05, 0.05, 0.05]),
    )

In this simple example we model a single point source, which might correspond to one lensed quasar or supernovae.
However, **PyAutoLens** supports model-fits to datasets with many lensed point-sources, for example in galaxy clusters.

Each point source dataset is therefore passed into a ``PointSourceDict`` object before the model-fit is performed. For
this simple example only one dataset is passed in, but in the galaxy-cluster examples you'll see this object makes it
straightforward to model datasets with many lensed sources.

.. code-block:: bash

    point_source_dict = al.PointSourceDict(point_source_dataset_list=[point_source_dataset])


We can print the ``positions`` of this dictionary and dataset, as well as their noise-map values.

.. code-block:: bash

    print("Point Source Dataset Name:")
    print(point_source_dict["point_0"].name)
    print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
    print(point_source_dict["point_0"].positions.in_list)
    print("Point Source Multiple Image Noise-map Values:")
    print(point_source_dict["point_0"].positions_noise_map.in_list)

Every point-source dataset in the ``PointSourceDict`` has a name, which in this example was ``point_0``. This ``name``
pairs the dataset to the ``PointSource`` in the model below. Because the name of the dataset is ``point_0``, the
only ``PointSource`` object that is used to fit it must have the name ``point_0``.

This ensures if a dataset has many point sources (e.g. galaxy clusters) it is clear how the model pairs the data.

It is straight forward to fit a lens model to a point source dataset, using the same API that we saw for imaging and
interferometer datasets.

This uses an ``AnalysisPointSource`` object which fits the lens model in the correct way for a point source dataset.
This includes mapping the ``name``'s of each dataset in the ``PointSourceDict`` to the names of the point sources in
the lens model.

.. code-block:: bash

    lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.PointSource)

    model = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)

    search = af.DynestyStatic(name="overview_interferometer")

    analysis = al.AnalysisInterferometer(dataset=interferometer)

    result = search.fit(model=model, analysis=analysis)