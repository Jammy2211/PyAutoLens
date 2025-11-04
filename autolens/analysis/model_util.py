import numpy as np
from typing import Optional, Tuple

import autofit as af
import autolens as al


def mge_model_from(
    mask_radius: float,
    total_gaussians: int = 30,
    gaussian_per_basis: int = 1,
    centre_prior_is_uniform: bool = True,
    centre: Tuple[float, float] = (0.0, 0.0),
    centre_fixed: Optional[Tuple[float, float]] = None,
    use_spherical: bool = False,
) -> af.Collection:
    """
    Construct a Multi-Gaussian Expansion (MGE) for the lens or source galaxy light

    This model is designed as a "start here" configuration for lens modeling:

    - The lens and source light are represented by a Basis object composed of many
      Gaussian light profiles with fixed logarithmically spaced widths (`sigma`).
    - All Gaussians within each basis share common centres and ellipticity
      components, reducing degeneracy while retaining flexibility.

    - Users can combine with a lens mass model of their choiuce.

    The resulting model provides a good balance of speed, flexibility, and accuracy
    for fitting most galaxy-scale strong lenses.

    This code is mostly to make the API simple for new users, hiding the technical
    details of setting up an MGE. More advanced users may wish to customize the
    model further.

    Parameters
    ----------
    mask_radius
        The outer radius (in arcseconds) of the circular mask applied to the data.
        This determines the maximum Gaussian width (`sigma`) used in the lens MGE.
    lens_total_gaussians
        Total number of Gaussian light profiles used in the lens MGE basis.
    source_total_gaussians
        Total number of Gaussian light profiles used in the source MGE basis.
    lens_gaussian_per_basis
        Number of separate Gaussian bases to include for the lens light profile.
        Each basis has `lens_total_gaussians` components.
    source_gaussian_per_basis
        Number of separate Gaussian bases to include for the source light profile.
        Each basis has `source_total_gaussians` components.

    Returns
    -------
    model : af.Collection
        An `autofit.Collection` containing:
        - A lens galaxy at redshift 0.5, with:
          * bulge light profile: MGE basis of Gaussians
          * mass profile: Isothermal ellipsoid
          * external shear
        - A source galaxy at redshift 1.0, with:
          * bulge light profile: MGE basis of Gaussians

    Notes
    -----
    - Lens light Gaussians have widths (sigma) logarithmically spaced between 0.01"
      and the mask radius.
    - Source light Gaussians have widths logarithmically spaced between 0.01" and 1.0".
    - Gaussian centres are free parameters but tied across all components in each
      basis to reduce dimensionality.
    - This function is a convenience utility: it hides the technical setup of MGE
      composition and provides a ready-to-use lens model for quick experimentation.
    """

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

    # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

    if centre_fixed is not None:
        centre_0 = centre[0]
        centre_1 = centre[1]
    elif centre_prior_is_uniform:
        centre_0 = af.UniformPrior(
            lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1
        )
        centre_1 = af.UniformPrior(
            lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1
        )
    else:
        centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.3)
        centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.3)

    if use_spherical:
        model_cls = al.lp_linear.GaussianSph
    else:
        model_cls = al.lp_linear.Gaussian

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        # A list of Gaussian model components whose parameters are customized belows.

        gaussian_list = af.Collection(
            af.Model(model_cls) for _ in range(total_gaussians)
        )

        # Iterate over every Gaussian and customize its parameters.

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
            gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
            if not use_spherical:
                gaussian.ell_comps = gaussian_list[
                    0
                ].ell_comps  # All Gaussians have same elliptical components.
            gaussian.sigma = (
                10 ** log10_sigma_list[i]
            )  # All Gaussian sigmas are fixed to values above.

        bulge_gaussian_list += gaussian_list

    # The Basis object groups many light profiles together into a single model component.

    return af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )


def simulator_start_here_model_from(
    include_lens_light: bool = True, use_point_source: bool = False
):

    if include_lens_light:
        bulge = af.Model(al.lp_snr.Sersic)

        bulge.centre = (0.0, 0.0)
        bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.signal_to_noise_ratio = af.UniformPrior(
            lower_limit=20.0, upper_limit=60.0
        )
        bulge.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)
        bulge.sersic_index = af.TruncatedGaussianPrior(
            mean=4.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
        )
    else:
        bulge = None

    mass = af.Model(al.mp.Isothermal)

    mass.centre = (0.0, 0.0)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.8)

    shear = af.Model(al.mp.ExternalShear)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    if use_point_source:

        point_0 = af.Model(al.ps.PointFlux)

        point_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        point_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
        point_0.flux = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    else:

        bulge = af.Model(al.lp_snr.Sersic)

        bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
        bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
        )
        bulge.signal_to_noise_ratio = af.UniformPrior(
            lower_limit=10.0, upper_limit=30.0
        )
        bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=3.0)
        bulge.sersic_index = af.TruncatedGaussianPrior(
            mean=2.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
        )

        source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    return lens, source
