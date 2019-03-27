from autolens.model.galaxy.summary import galaxy_summary

from test.unit.mock.mock_summary import MockGalaxy, MockTruncatedNFW

import os

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizeGalaxy:

    def test__summarize_galaxy_with_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        galaxy_summary.summarize_galaxy(summary_file=summary_file, galaxy=MockGalaxy(mass_profiles=[MockTruncatedNFW()]),
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