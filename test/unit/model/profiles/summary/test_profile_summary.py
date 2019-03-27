from autolens.model.profiles.summary import profile_summary

from test.unit.mock.mock_summary import MockMassProfile, MockNFW, MockTruncatedNFW

import pytest
import os
import shutil

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizeMassProfile:

    def test__summarize_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_mass_profile(summary_file=summary_file, mass_profile=MockTruncatedNFW(),
                                               critical_surface_mass_density=1.0, cosmic_average_mass_density_arcsec=1.0,
                                               radii=[10.0, 500.0])

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Mass Profile = MockTruncatedNFW \n'
        assert summary_text[1] == '\n'
        assert summary_text[2] == 'Einstein Radius = 10.00" \n'
        assert summary_text[3] == 'Mass within Einstein Radius = 1.0000e+03 solMass \n'
        assert summary_text[4] == 'Mass within 10.00" = 1.0000e+03 solMass \n'
        assert summary_text[5] == 'Mass within 500.00" = 1.0000e+03 solMass \n'
        assert summary_text[6] == 'Rho at scale radius = 100.00 \n'
        assert summary_text[7] == 'Delta concentration = 200.00 \n'
        assert summary_text[8] == 'Concentration = 300.00 \n'
        assert summary_text[9] == 'Radius at 200x cosmic average density = 400.00" \n'
        assert summary_text[10] == 'Mass at 200x cosmic average density = 500.00 solMass \n'
        assert summary_text[11] == 'Mass at truncation radius = 600.00 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

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

    def test__summarize_truncated_nfw_challenge_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_truncated_nfw_challenge_mass_profile(summary_file=summary_file,
                                                                       truncated_nfw_challenge=MockTruncatedNFW())

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