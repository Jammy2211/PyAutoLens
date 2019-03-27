from autolens.lens.summary import plane_summary

from test.unit.mock.mock_summary import MockPlane, MockGalaxy, MockTruncatedNFW

import os

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizePlane:

    def test__summarize_plane_with_galaxy_with_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        mock_galaxy = MockGalaxy(mass_profiles=[MockTruncatedNFW()])

        plane_summary.summarize_plane(summary_file=summary_file, plane=MockPlane(galaxies=[mock_galaxy]),
                                        critical_surface_mass_density=1.0, radii=[10.0, 500.0])

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text[0] == 'Plane Redshift = 0.5 \n'
        assert summary_text[1] == '\n'
        assert summary_text[2] == 'Galaxy = lol \n'
        assert summary_text[3] == '\n'
        assert summary_text[4] == 'Mass Profile = MockTruncatedNFW \n'
        assert summary_text[5] == '\n'
        assert summary_text[6] == 'Einstein Radius = 10.00" \n'
        assert summary_text[7] == 'Mass within Einstein Radius = 1.0000e+03 solMass \n'
        assert summary_text[8] == 'Mass within 10.00" = 1.0000e+03 solMass \n'
        assert summary_text[9] == 'Mass within 500.00" = 1.0000e+03 solMass \n'
        assert summary_text[10] == 'Rho at scale radius = 100.00 \n'
        assert summary_text[11] == 'Delta concentration = 200.00 \n'
        assert summary_text[12] == 'Concentration = 300.00 \n'
        assert summary_text[13] == 'Radius at 200x cosmic average density = 400.00" \n'
        assert summary_text[14] == 'Mass at 200x cosmic average density = 500.00 solMass \n'
        assert summary_text[15] == 'Mass at truncation radius = 600.00 solMass \n'

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')