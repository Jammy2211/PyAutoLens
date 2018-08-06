import getdist
import re
import os
import math
import numpy as np
from astropy import cosmology
import matplotlib.pyplot as plt

from autolens.analysis import galaxy
from autolens.profiles import mass_profiles

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def getModelFolders(dir):
    regex = re.compile('.*Hyper')
    folders = [name for name in os.listdir(dir) if os.path.isdir(dir+name) and re.match(regex, name) == None]
    return folders

class SLACS(object):
            
    def __init__(self, image_dir, image_name):

        self.image_dir = image_dir
        self.image_name = image_name

        cosmological_model = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

        if image_name == 'SLACSJ0252+0039':

            self.redshift = 0.2803
            self.source_redshift = 0.9818
            self.einstein_radius = 1.02
            self.source_light_min = 0.8
            self.source_light_max = 1.3
            self.ein_r_kpc = 4.18
            self.mass = '{0:e}'.format(10 ** 11.25)
            self.radius = 4.4
            self.chab_stellar_mass = '{0:e}'.format(10 ** 11.21)
            self.chab_stellar_mass_lower = '{0:e}'.format(10 ** 11.08)
            self.chab_stellar_mass_upper = '{0:e}'.format(10 ** 11.34)
            self.sal_stellar_mass = '{0:e}'.format(10 ** 11.46)
            self.sal_stellar_mass_lower = '{0:e}'.format(10 ** 11.33)
            self.sal_stellar_mass_upper = '{0:e}'.format(10 ** 11.59)
            self.chab_stellar_frac = 0.4
            self.chab_stellar_frac_error = 0.12
            self.sal_stellar_frac = 0.71
            self.sal_stellar_frac_error = 0.21
            
        elif image_name == 'SLACSJ1250+0523':

            self.redshift = 0.2318
            self.source_redshift = 0.7953
            self.einstein_radius = 1.120
            self.source_light_min = 0.5
            self.source_light_max = 1.5
            self.ein_r_kpc = 4.18
            self.mass = '{0:e}'.format(10 ** 11.26)
            self.chab_stellar_mass = '{0:e}'.format(10 ** 11.53)
            self.chab_stellar_mass_lower = '{0:e}'.format(10 ** 11.46)
            self.chab_stellar_mass_upper = '{0:e}'.format(10 ** 11.60)
            self.sal_stellar_mass = '{0:e}'.format(10 ** 11.77)
            self.sal_stellar_mass_lower = '{0:e}'.format(10 ** 11.7)
            self.sal_stellar_mass_upper = '{0:e}'.format(10 ** 12.84)
            self.chab_stellar_frac = 0.68
            self.chab_stellar_frac_error = 0.11
            self.sal_stellar_frac = 1.2
            self.sal_stellar_frac_error = 0.19

        elif image_name == 'SLACSJ1430+4105':

            self.redshift = 0.2850
            self.source_redshift = 0.5753
            self.einstein_radius = 1.468
            self.source_light_min = 0.2
            self.source_light_max = 2.3
            self.our_einr_arc = 1.4668492852
            self.ein_r_kpc = 6.53
            self.mass = '{0:e}'.format(10 ** 11.73)
            self.chab_stellar_mass = '{0:e}'.format(10 ** 11.68)
            self.chab_stellar_mass_lower = '{0:e}'.format(10 ** 11.56)
            self.chab_stellar_mass_upper = '{0:e}'.format(10 ** 11.8)
            self.sal_stellar_mass = '{0:e}'.format(10 ** 11.93)
            self.sal_stellar_mass_lower = '{0:e}'.format(10 ** 11.82)
            self.sal_stellar_mass_upper = '{0:e}'.format(10 ** 12.04)
            self.chab_stellar_frac = 0.33
            self.chab_stellar_frac_error = 0.09
            self.sal_stellar_frac = 0.59
            self.sal_stellar_frac_error = 0.16

        self.arcsec_per_kpc = cosmological_model.arcsec_per_kpc_proper(self.redshift).value
        self.kpc_per_arcsec = 1.0/self.arcsec_per_kpc

        print(self.kpc_per_arcsec)
        # stop

    def load_samples(self, pdf_file, center_skip, ltm_skip):

        self.pdf = getdist.mcsamples.loadMCSamples(pdf_file)
        self.params = self.pdf.paramNames
        
        self.center_skip = center_skip
        self.ltm_skip = ltm_skip

    def setup_lensing_plane(self, values):

        sersic_bulge = mass_profiles.EllipticalSersicMass(centre=(values[0], values[1]), axis_ratio=values[5],
                                                          phi=values[6], intensity=values[2],
                                                          effective_radius=values[3],
                                                          sersic_index=values[4],
                                                          mass_to_light_ratio=values[12 + self.center_skip])

        exponential_halo = mass_profiles.EllipticalExponentialMass(axis_ratio=values[9 + self.center_skip],
                                                                   phi=values[10 + self.center_skip],
                                                                   intensity=values[7 + self.center_skip],
                                                                   effective_radius=values[8 + self.center_skip],
                                                                   mass_to_light_ratio=values[
                                                                    12 + self.ltm_skip + self.center_skip])

        dark_matter_halo = mass_profiles.SphericalNFW(kappa_s=values[11 + self.center_skip],
                                                      scale_radius=30.0*self.arcsec_per_kpc)

        self.lens_galaxy = galaxy.Galaxy(redshift=self.redshift,
                                         mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

        self.source_galaxy = galaxy.Galaxy(redshift=self.source_redshift)

    def setup_lensing_plane_ltm2r(self, values):

        sersic_bulge = mass_profiles.EllipticalSersicMass(centre=(values[0], values[1]), axis_ratio=values[5],
                                                          phi=values[6], intensity=values[2],
                                                          effective_radius=values[3],
                                                          sersic_index=values[4],
                                                          mass_to_light_ratio=values[12 + self.center_skip])

        exponential_halo = mass_profiles.EllipticalSersicMassRadialGradient(axis_ratio=values[9 + self.center_skip],
                                                                   phi=values[10 + self.center_skip],
                                                                   intensity=values[7 + self.center_skip],
                                                                   effective_radius=values[8 + self.center_skip],
                                                                   sersic_index=1.0,
                                                                   mass_to_light_ratio=values[
                                                                    12 + self.ltm_skip + self.center_skip],
                                                                   mass_to_light_gradient=values[
                                                                    13 + self.ltm_skip + self.center_skip])

        dark_matter_halo = mass_profiles.SphericalNFW(kappa_s=values[11 + self.center_skip],
                                                      scale_radius=30.0*self.arcsec_per_kpc)

        self.lens_galaxy = galaxy.Galaxy(redshift=self.redshift,
                                         mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

        self.source_galaxy = galaxy.Galaxy(redshift=self.source_redshift)

    def setup_model(self, model_index, ltmr2=False):

        values = self.pdf.samples[model_index]

    def masses_of_all_samples(self, radius_kpc):

        model_indexes = []
        sample_weights = []
        total_masses = []
        stellar_masses = []
        dark_masses = []
        stellar_fractions = []

        for i in range(len(self.pdf.samples)):

            if self.pdf.weights[i] > 1e-6:

                model_indexes.append(i)

                values = self.pdf.samples[i]

                self.setup_lensing_plane(values)

                sample_weights.append(self.pdf.weights[i])

                radius_arc = radius_kpc * self.arcsec_per_kpc
                masses = self.lens_galaxy.dimensionless_mass_within_circles_individual(radius=radius_arc)

                total_mass = sum(masses)
                stellar_mass = (masses[0] + masses[1])
                dark_mass = masses[2]

                total_masses.append(total_mass)
                stellar_masses.append(stellar_mass)
                dark_masses.append(dark_mass)
                stellar_fractions.append(stellar_mass/total_mass)

        return model_indexes, sample_weights, total_masses, stellar_masses, dark_masses, stellar_fractions

    def weighted_densities_vs_radii(self, radius_kpc, weight_cut=1e-6, number_bins=60):

        values = self.pdf.samples[1]
        self.setup_lensing_plane(values)
        radii = list(np.linspace(5e-3, radius_kpc*self.arcsec_per_kpc, number_bins+1))

        self.bulge_density_plot = []
        self.bulge_density_upper_plot = []
        self.bulge_density_lower_plot = []

        self.halo_density_plot = []
        self.halo_density_upper_plot = []
        self.halo_density_lower_plot = []

        self.dark_density_plot = []
        self.dark_density_upper_plot = []
        self.dark_density_lower_plot = []

        self.radii_plot = []

        for r in range(number_bins):

            annuli_area = (math.pi * radii[r + 1] ** 2 - math.pi * radii[r] ** 2)

            weights = []
            bulge_densities = []
            halo_densities = []
            dark_densities = []

            for i in range(len(self.pdf.samples)):

                if self.pdf.weights[i] > weight_cut:

                    weights.append(self.pdf.weights[i])
                    values = self.pdf.samples[i]
                    self.setup_lensing_plane(values)

                    densities = ((self.lens_galaxy.dimensionless_mass_within_circles_individual(radii[r + 1]) -
                                  self.lens_galaxy.dimensionless_mass_within_circles_individual(radii[r])) / annuli_area)

                    bulge_densities.append(densities[0])
                    halo_densities.append(densities[1])
                    dark_densities.append(densities[2])


            bulge_avg, bulge_std = weighted_avg_and_std(np.asarray(bulge_densities), np.asarray(weights))
            halo_avg, halo_std = weighted_avg_and_std(np.asarray(halo_densities), np.asarray(weights))
            dark_avg, dark_std = weighted_avg_and_std(np.asarray(dark_densities), np.asarray(weights))

            self.bulge_density_plot.append(bulge_avg)
            self.bulge_density_upper_plot.append(bulge_avg + bulge_std)
            self.bulge_density_lower_plot.append(bulge_avg - bulge_std)

            self.halo_density_plot.append(halo_avg)
            self.halo_density_upper_plot.append(halo_avg + halo_std)
            self.halo_density_lower_plot.append(halo_avg - halo_std)

            self.dark_density_plot.append(dark_avg)
            self.dark_density_upper_plot.append(dark_avg + dark_std)
            self.dark_density_lower_plot.append(dark_avg - dark_std)

            self.radii_plot.append( ((radii[r+1] + radii[r]) / 2.0) * self.kpc_per_arcsec)

    def plot_density(self, image_name='', labels=None, xaxis_is_physical=True, yaxis_is_physical=True):

        plt.title('Decomposed surface density profile of ' + image_name, size=20)

        if xaxis_is_physical:
            plt.xlabel('Distance From Galaxy Center (kpc)', size=20)
        else:
            plt.xlabel('Distance From Galaxy Center (")', size=20)

        if yaxis_is_physical:
            plt.ylabel(r'Surface Mass Density $\Sigma$ ($M_{\odot}$ / kpc $^2$)', size=20)
        else:
            pass

        plt.semilogy(self.radii_plot, self.bulge_density_plot, color='r', label='Sersic Bulge')
        plt.semilogy(self.radii_plot, self.halo_density_plot, color='g', label='EllipticalExponentialMass Halo')
        plt.semilogy(self.radii_plot, self.dark_density_plot, color='k', label='Dark Matter Halo')

        plt.semilogy(self.radii_plot, self.bulge_density_upper_plot, color='r', linestyle='--')
        plt.semilogy(self.radii_plot, self.halo_density_upper_plot, color='g', linestyle='--')
        plt.semilogy(self.radii_plot, self.dark_density_upper_plot, color='k', linestyle='--')

        plt.semilogy(self.radii_plot, self.bulge_density_lower_plot, color='r', linestyle='--')
        plt.semilogy(self.radii_plot, self.halo_density_lower_plot, color='g', linestyle='--')
        plt.semilogy(self.radii_plot, self.dark_density_lower_plot, color='k', linestyle='--')

        # plt.loglog(radii_plot, density_0, color='r', label='Sersic Bulge')
        # plt.loglog(radii_plot, density_1, color='g', label='EllipticalExponentialMass Halo')
        # plt.loglog(radii_plot, density_2, color='k', label='Dark Matter Halo')

        # plt.semilogy(radii_plot, density_lower[:, 0], color='r', linestyle='--')
        # plt.semilogy(radii_plot, density_upper[:, 0], color='r', linestyle='--')
        # plt.semilogy(radii_plot, density_lower[:, 1], color='g', linestyle='--')
        # plt.semilogy(radii_plot, density_upper[:, 1], color='g', linestyle='--')
        # plt.semilogy(radii_plot, density_lower[:, 2], color='k', linestyle='--')
        # plt.semilogy(radii_plot, density_upper[:, 2], color='k', linestyle='--')

        plt.axvline(x=self.einstein_radius*self.kpc_per_arcsec, linestyle='--')
        plt.axvline(x=self.source_light_min*self.kpc_per_arcsec, linestyle='-')
        plt.axvline(x=self.source_light_max*self.kpc_per_arcsec, linestyle='-')

        plt.legend()
        plt.show()