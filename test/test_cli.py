import subprocess
from os import path


class TestCLI(object):
    def test_cli_hdu(self):
        directory = path.dirname(path.realpath(__file__))

        result = subprocess.check_output(
            "python {0}/../autolens pipeline test "
            "--data={0}/../data/slacs03_all/slacs_1_post.fits "
            "--image-hdu=1 "
            "--noise-hdu=2 "
            "--psf-hdu=3 "
            "--pixel-scale=0.05".format(directory),
            shell=True)

        assert result is not None

    def test_cli_files(self):
        directory = path.dirname(path.realpath(__file__))

        data_directory = "{}/integration/data/l2g".format(directory)

        result = subprocess.check_output(
            "python {0}/../autolens pipeline test "
            "--image={1}/image.fits "
            "--noise={1}/noise.fits "
            "--psf={1}/psf.fits "
            "--pixel-scale=0.05".format(directory, data_directory),
            shell=True)

        assert result is not None
