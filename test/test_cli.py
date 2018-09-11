import subprocess
from os import path


class TestCLI(object):
    def test_cli(self):
        directory = path.dirname(path.realpath(__file__))

        result = subprocess.check_output(
            "python {0}/../autolens pipeline test "
            "--data={0}/../data/slacs03_all/slacs_1_post.fits "
            "--image-hdu=1 --noise-hdu=2 "
            "--psf-hdu=3 "
            "--pixel-scale=0.05".format(directory),
            shell=True)

        print(result)
