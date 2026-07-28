---
title: "PyAutoLens-JAX: Differentiable GPU-accelerated strong and weak lensing from galaxies to clusters"
tags:
  - Python
  - astronomy
  - gravitational lensing
  - automatic differentiation
  - GPU acceleration
authors:
  - name: James W. Nightingale
    orcid: 0000-0002-8987-7401
    affiliation: 1
    corresponding: true
affiliations:
  - name: Institute for Computational Cosmology, Durham University, United Kingdom
    index: 1
date: 14 July 2026
bibliography: paper.bib
---

# Summary

Gravitational lensing probes luminous and dark matter in galaxy-, group-, and cluster-scale systems. Lensing datasets 
are growing rapidly in size: Stage IV surveys such as Euclid [@EuclidCollaboration2025] and the Vera C. Rubin 
Observatory [@LSSTDarkEnergyScienceCollaboration2012] will measure billions of galaxies across large fractions of the 
sky. They are also growing in diversity: a single system may include strong- and weak-lensing constraints, multi-band 
optical and infrared imaging, radio-interferometric visibilities, and point-source measurements of lensed quasars or
supernovae. Fully exploiting these observations requires joint probabilistic modelling across lensing scales and data types. 
However, the increasing volume of data and complexity of lens models make these analyses increasingly computationally expensive.

PyAutoLens is now implemented using JAX throughout its core modelling framework, providing just-in-time compilation, 
GPU acceleration, and automatic differentiation without introducing a separate package or replacing its established 
object-oriented API. Galaxy-, group-, and cluster-scale lens models can be constrained using CCD imaging, 
interferometer visibilities, point-source observables, and weak-lensing catalogues fully in JAX. Crucially, 
these are not isolated capabilities: users can combine multiple datasets and strong- and weak-lensing constraints 
within a single differentiable, GPU-accelerated probabilistic model. In doing so, PyAutoLens-JAX allows 
gravitational-lensing analyses to scale with the size and complexity of next-generation datasets.

# Statement of need

Modern lensing analyses must handle both rapidly growing samples and far more information for each system. 
High-resolution imaging constrains extended arcs and lens-galaxy light; interferometric observations probe 
source structure in the visibility domain; strongly lensed variable and transient sources, including quasars and 
supernovae, provide image positions and time delays; and weak lensing traces mass on larger spatial scales. 
Group and cluster lenses add multiple deflectors, source planes, and complex mass distributions. Joint modelling 
can break degeneracies and yield more complete physical constraints, but it also increases model dimensionality and 
the cost of each likelihood evaluation. Providing the required lensing calculations for all these datasets within a 
single GPU-accelerated, automatically differentiable package therefore offers a compelling route to making the
science these data enable feasible.

Conventional derivative-free inference becomes difficult as analyses combine pixelized source reconstructions, 
multi-band datasets, millions of interferometer visibilities, strong- and weak-lensing constraints, and multi-scale
mass models across increasingly large lens samples. PyAutoLens-JAX addresses this computational bottleneck by 
making the complete modelling framework compatible with GPU execution and automatic differentiation. This enables faster likelihood evaluation and the use of gradient-based optimisation and sampling methods across the full range of PyAutoLens datasets and lensing regimes.

# State of the field

<!--
Position PyAutoLens-JAX relative to the published PyAutoLens software
[@Nightingale2021] and other current strong- and weak-lensing packages. Focus on
the combination of established modelling abstractions, joint multi-dataset
inference, automatic differentiation, and GPU execution.
-->

# Software design

## Differentiable modelling

<!--
Describe the JAX-compatible numerical backend, just-in-time compilation,
automatic differentiation, and preservation of the public PyAutoLens API. Cite
JAX [@jax2018github] and explain which model components are differentiable.
-->

## Joint datasets and lensing regimes

<!--
Describe how analyses combine imaging, interferometer, point-source, and weak-
lensing observations across lens planes and physical scales in one likelihood.
Include a concise representative example or schematic.
-->

## End-to-end modelling benchmarks

All benchmarks use gradient-based inference with JAX automatic differentiation and are executed on an NVIDIA A100 GPU. Unless otherwise stated, the lens model comprises a singular isothermal ellipsoid with external shear, a multi-Gaussian expansion for the lens light, and a pixelized reconstruction of the lensed source. For every analysis, we report the total wall-clock time, JAX compilation time, number of likelihood evaluations, and post-compilation runtime.

### Single-dataset and single-regime benchmarks

These benchmarks establish GPU acceleration and automatic differentiation across the individual lensing scales and observational data types supported by PyAutoLens-JAX.

- **Galaxy-scale CCD imaging:** Model JWST COSMOS-Web Ring F150W imaging, including lens-light subtraction and a pixelized source reconstruction, in approximately five minutes.
- **Interferometry:** Model a real ALMA strong-lensing dataset containing more than one million interferometer visibilities in approximately five minutes.
- **Point-source lensing:** Model a real multiply imaged quasar or supernova using point-source observables, including image positions and, where available, time delays or flux information, in under five minutes.
- **Group-scale strong lensing:** Model a real group-scale lens containing multiple deflecting galaxies in under five minutes, demonstrating that PyAutoLens-JAX is not restricted to isolated galaxy-scale lenses.
- **Cluster-scale strong lensing:** Model a real cluster lens with multiple mass components, multiple images, and potentially multiple source planes in under five minutes.
- **Weak lensing:** Fit a weak-lensing shear catalogue using a differentiable JAX likelihood in under five minutes, demonstrating that PyAutoLens-JAX is not restricted to strong-lensing data.

### Joint and multi-dataset benchmarks

These benchmarks demonstrate the central capability of PyAutoLens-JAX: different datasets, lensing regimes, and physical scales can be combined within a single differentiable, GPU-accelerated probabilistic model.

- **Multi-band imaging:** Jointly model the four available JWST COSMOS-Web Ring bands, constraining a common lens mass model while fitting the wavelength-dependent lens and source emission in each dataset.
- **Joint strong and weak lensing:** Constrain a single group- or cluster-scale mass model using both strong-lensing and weak-lensing observables.
- **Imaging and point-source lensing:** Jointly model extended arcs and point-source constraints from a lensed quasar or supernova within the same lens model.
- **Imaging and interferometry:** Jointly fit optical or infrared imaging and radio or submillimetre interferometer visibilities, constraining a common mass model using complementary observations of the lensed source.

The single examples verify that each major PyAutoLens likelihood and modelling regime is JAX compatible. The combined examples then demonstrate that these capabilities are not implemented as isolated workflows: they can be composed while retaining a common object-oriented API, automatic differentiation, and GPU execution.

# Research impact statement

<!--
Give specific evidence of research enabled by PyAutoLens and the JAX framework:
published or ongoing projects, external adoption, survey-scale applications,
and reproducible examples. Keep this about software impact rather than new
scientific results.
-->

# AI usage disclosure

Generative AI tools were used to scaffold this manuscript template and may be used to assist drafting. All scientific claims, citations, and prose are reviewed and verified by the authors.

# Acknowledgements

<!-- List funding, facilities, software contributors, and other support. -->

# References
