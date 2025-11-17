.. _overview_2_new_user_guide:

New User Guide
==============

**PyAutoLens** is an extensive piece of software with functionality for doing many different analysis tasks, fitting
different data types and it is used for a variety of different science cases. This means the documentation is quite
extensive, and it may initially be difficult to find the example script you need.

The ``autolens_workspace`` has five ``start_here.ipynb`` notebooks, and you need to determine which is most relevant
to your scientific interests.

You can access the notebooks by clicking on the embedded links below:

 - `imaging/start_here.ipynb <https://colab.research.google.com/github/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/start_here.ipynb>`_: Galaxy scale strong lenses observed with CCD imaging (e.g. Hubble, James Webb).
 - `interferometer/start_here.ipynb <https://colab.research.google.com/github/Jammy2211/autolens_workspace/blob/release/notebooks/interferometer/start_here.ipynb>`_: Galaxy scale strong lenses observed with interferometer data (e.g. ALMA).
 - `point_source/start_here.ipynb <https://colab.research.google.com/github/Jammy2211/autolens_workspace/blob/release/notebooks/point_source/start_here.ipynb>`_: Galaxy scale strong lenses with a lensed point source (e.g. lensed quasars).
 - `group/start_here.ipynb <https://colab.research.google.com/github/Jammy2211/autolens_workspace/blob/release/notebooks/group/start_here.ipynb>`_: Group scale strong lenses where there are 2-10 lens galaxies.
 - `cluster/start_here.ipynb <https://colab.research.google.com/github/Jammy2211/autolens_workspace/blob/release/notebooks/cluster/start_here.ipynb>`_: Cluster scale strong lenses with 2+ lenses and 5+ source galaxies.

If you are unsure based on the brief descriptions above, answer the following two questions to work out where to start.

What Scale Lens?
----------------

What size and scale of strong lens system are you expecting to work with?

There are three scales to choose from:

- **Galaxy Scale**: Made up of a single lens galaxy lensing a single source galaxy, the simplest strong lens you can get!
  If you're interested in galaxy scale lenses, go to the question below called "What Data Type?".

- **Group Scale**: Strong Lens Groups contains 2-10 lens galaxies, normally with one main large galaxy responsible for the majority of lensing.
  They also typically lens just one source galaxy. If you are interested in groups, go to the ``group/start_here.ipynb`` notebook.

- **Cluster Scale**: Strong Lens Galaxy clusters often contained 20-50, or more, lens galaxies, lensing 10, or more, sources galaxies.
  If you are interested in clusters, go to the ``cluster/start_here.ipynb`` notebook.

What Dataset Type?
------------------

If you are interested in galaxy-scale strong lenses, you now need to decide what type of strong lens data you are
interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to ``imaging/start_here.ipynb``.

- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to ``interferometer/start_here.ipynb``.

- **Point Sources**: For strongly lensed point sources (e.g. lensed quasars, supernovae), go to ``point_source/start_here.ipynb``.

Still Unsure?
-------------

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task.
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you
should just read through a few different notebooks and go from there.

HowToLens
---------

For experienced scientists, the **PyAutoLens** examples will be simple to follow. Concepts surrounding strong lensing may
already be familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToLens** Jupyter Notebook lectures provide exactly this. They are a 3+ chapter guide which thoroughly
take you through the core concepts of strong lensing, teach you the principles of the statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

To complete thoroughly, they'll probably take 2-4 days, so you may want try moving ahead to the examples but can
go back to these lectures if you find them hard to follow.

If this sounds like it suits you, checkout the ``autolens_workspace/notebooks/howtolens`` package now.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/howtolens

Wrap Up
-------

After completing this guide, you should be able to use **PyAutoLens** for your science research.

The biggest decisions you'll need to make are what features and functionality your specific science case requires,
which the next readthedocs page gives an overview of to help you decide.