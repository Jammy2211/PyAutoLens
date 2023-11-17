.. _overview_9_clusters:

Cluster-Scale Lenses
====================

Galaxy clusters are the beasts of strong lensing. They contain tens or hundreds of lens galaxies and lensed sources,
with lensed sources at many different redshifts requiring full multi-plane ray-tracing calculations. They contain one
or more brightest cluster galaxy(s) a large scale dark matter halo and have arcs with Einstein Radii 10.0" -> 100.0"
and beyond.

Here is an image of SDSS1152P3312, the example cluster we will adopt for illustrating
**PyAutoLens**'s cluster modeling tools:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/clusters/cluster.png?raw=true
  :width: 600
  :alt: Alternative text

Point Source
------------

Just like for group-scale lenses, we will fit the cluster using a point-source dataset.

.. code-block:: python

    point_dict = al.PointDict.from_json(
        file_path=path.join(dataset_path, "point_dict.json")
    )

Source-Plane Chi Squared
------------------------

To model a cluster, we assume that every source galaxy is a ``PointSrcChi`` model, which means the goodness-of-fit is
evaluated in the source-plane. This removes the need to iteratively solve the lens equation. However, we still define
a ``PointSolver``, in case we wish to perform image-plane fits.

.. code-block:: python

    point_solver = al.PointSolver(grid=grid, pixel_scale_precision=0.025)

Lens Model
----------

A cluster scale strong lens model is typically composed of the following:

 - One or more brightest cluster galaxies (BCG), which are sufficiently large that we model them individually.

 - One or more cluster-scale dark matter halos, which are again modeled individually.

 - Tens or hundreds of galaxy cluster member galaxies. The low individual masses of these objects means we cannot
  model them individually are constrain their mass, but their collectively large enough mass to need modeling. These
  are modeled using a scaling relation which assumes that light traces mass, where the luminosity of each individual
  galaxy is used to set up this scaling relation.

 - Tens or hundreds of source galaxies, each with multiple sets of images that constrain the lens model. These are
 modeled as a point-source, although **PyAutoLens** includes tools for modeling the imaging data of sources once a good
 lens model is inferred. The source redshifts are also used to account for multi-plane ray-tracing.

Therefore, we again load the model from a ``.json`` file:

.. code-block:: python

    model_path = path.join("scripts", "group", "models")
    model_file = path.join(model_path, "lens_x3__source_x1.json")

    lenses_file = path.join(model_path, "lenses.json")
    lenses = af.Collection.from_json(file=lenses_file)

    sources_file = path.join(model_path, "sources.json")
    sources = af.Collection.from_json(file=sources_file)

    galaxies = lenses + sources

    model = af.Collection(galaxies=galaxies)

SExtractor Catalogues
---------------------

Composing the lens model for cluster scale objects requires care, given there are could be hundreds of lenses and
sources galaxies. Manually writing the model in a Python script, in the way we do for galaxy-scale lenses, is therefore
not feasible.

For this cluster, we therefore composed the model by interfacing with Source Extractor
(https://sextract.readthedocs.io/) catalogue files. A full illustration of how to make the lens and source models
from catalogue files is given at the following links:

 `lenses <https://github.com/Jammy2211/autolens_workspace/blob/main/scripts/cluster/model_maker/example__lenses.py>`_
 `sources <https://github.com/Jammy2211/autolens_workspace/blob/main/scripts/cluster/model_maker/example__sources.py>`_

These files can be easily altered to compose a cluster model suited to your lens
dataset!

Lens Modeling
-------------

We are now able to model this dataset as a point source:

.. code-block:: python

    search = af.Nautilus(name="overview_clusters")

    analysis = al.AnalysisPoint(point_dict=point_dict, solver=point_solver)

    result = search.fit(model=model, analysis=analysis)

Result
------

The result contains information on the BCG, cluster scale dark matter halo and mass-light scaling relation:

.. code-block:: python

    print(result.max_log_likelihood_instance.galaxies.bcg.mass)
    print(result.max_log_likelihood_instance.galaxies.dark.mass)
    print(result.max_log_likelihood_instance.galaxies.scaling_relation)

Extended Source Fitting
-----------------------

For clsuter-scale lenses fitting the extended surface-brightness is extremely difficult. The models become high
dimensional and difficult to fit, and it becomes very computationally. Furthermore, the complexity of cluster mass
models can make it challenging to compose a mass model which is sufficiently accurate that a source reconstruction is
even feasible!

Nevertheless, we are currently developing tools that try and make this possible. These will take approaches like
fitting individual sources after modeling the entire cluster as a point-source and parallelizing the model-fitting
process out in a way that 'breaks-up' the model-fitting procedure.

These tools are in-development, but we are keen to have users with real sciences cases trial them as we develop
them. If you are interested please contact me! (https://github.com/Jammy2211).

Wrap-Up
-------

The `clusters <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/clusters>`_ package of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_  contains numerous example scripts for performing cluster-sale
modeling and simulating cluster-scale strong lens datasets.

