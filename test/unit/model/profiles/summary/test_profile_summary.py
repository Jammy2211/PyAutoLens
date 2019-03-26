from autolens.model.profiles.summary import profile_summary

import pytest
import os
import shutil

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class MockMassProfile(object):

    def __init__(self):
        pass

    @property
    def einstein_radius(self):
        return 10.0

    def mass_within_circle_in_mass_units(self, radius, critical_surface_mass_density):
        return 1000.0

class MockNFW(MockMassProfile):

    def __init__(self):
        super(MockNFW, self).__init__()

    def rho_at_scale_radius(self, critical_surface_mass_density_arcsec):
        return 100.0

    def delta_concentration(self, critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec):
        return 200.0

    def concentration(self, critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec):
        return 300.0

    def radius_at_200(self, critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec):
        return 400.0

    def mass_at_200(self, critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec):
        return 500.0

class MockTruncatedNFW(MockNFW):

    def __init__(self):
        super(MockTruncatedNFW, self).__init__()

    def mass_at_truncation_radius(self, critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec):
        return 600.0


class TestSummarizeMassProfile:

    def test__summarize_einstein_radius_and_mass___multiple_radii_both_output(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_einstein_radius_and_mass(summary_file=summary_file, mass_profile=MockMassProfile(),
                                                           critical_surface_mass_density=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Einstein Radius = 10.00" \n'
        assert summary_text[1] == 'Mass within Einstein Radius = 1.0000e+03 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_mass_within_radii___multiple_radii_both_output(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_mass_within_radii(summary_file=summary_file, mass_profile=MockMassProfile(),
                                                    radii=[10.0, 500.0], critical_surface_mass_density=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Mass within 10.00" = 1.0000e+03 solMass \n'
        assert summary_text[1] == 'Mass within 500.00" = 1.0000e+03 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_nfw_mass_profile(summary_file=summary_file, nfw=MockNFW(),
                                                   critical_surface_mass_density_arcsec=1.0,
                                                   cosmic_average_mass_density_arcsec=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Rho at scale radius = 100.00 \n'
        assert summary_text[1] == 'Delta concentration = 200.00 \n'
        assert summary_text[2] == 'Concentration = 300.00 \n'
        assert summary_text[3] == 'Radius at 200x cosmic average density = 400.00" \n'
        assert summary_text[4] == 'Mass at 200x cosmic average density = 500.00 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_truncated_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_truncated_nfw_mass_profile(summary_file=summary_file, truncated_nfw=MockTruncatedNFW(),
                                                             critical_surface_mass_density_arcsec=1.0,
                                                             cosmic_average_mass_density_arcsec=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Rho at scale radius = 100.00 \n'
        assert summary_text[1] == 'Delta concentration = 200.00 \n'
        assert summary_text[2] == 'Concentration = 300.00 \n'
        assert summary_text[3] == 'Radius at 200x cosmic average density = 400.00" \n'
        assert summary_text[4] == 'Mass at 200x cosmic average density = 500.00 solMass \n'
        assert summary_text[5] == 'Mass at truncation radius = 600.00 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')