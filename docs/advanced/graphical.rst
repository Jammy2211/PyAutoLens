.. _hierarchical_models:

Graphical & Hierarchical Models
===============================

Graphical modeling allows us to compose a single model that is fitted to an entire lens dataset. This model includes
a specific lens model for every individual lens in the sample, but also has shared parameters between these
individual lens models.

An example of such a model might have Cosmological parameters (e.g. the Hubble constant) as a global parameter
which is fitted simultaneously to many lens datasets, each with their own unique mass model.

An extension to a graphical model is a **hierarchical model**. Here, the shared parameter(s) of the model do not have
exactly the same value in every dataset. Instead, the shared parameter(s) are drawn from a common parent
distribution (e.g. a Gaussian). It is the parameters of this parent distribution that are shared
across the dataset, and these are the parameters we ultimately wish to infer to understand the global behaviour of the
model.

An example of such a model might be determining the parent distribution from which the density slope of strong lens
galaxies are drawn.

A full description of graphical and hierarchical models can be found `in the graphical package of the autolens_workspace <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/imaging/advanced/graphical>`_.