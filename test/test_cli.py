# import subprocess
# from os import path
#
# # TODO : Dont know how to fix this.
#
# class TestCLI(object):
#     # def test_cli_hdu(self):
#     #     directory = path.dirname(path.realpath(__file__))
#     #
#     #     result = subprocess.check_output(
#     #         "python {0}/../autolens pipeline test "
#     #         "--datas={0}/../datas/slacs03_all/slacs_1_post.fits "
#     #         "--image-hdu=1 "
#     #         "--noise-hdu=2 "
#     #         "--psf-hdu=3 "
#     #         "--pixel-scale=0.05".format(directory),
#     #         shell=True)
#     #
#     #     assert result is not None
#
#     def test_cli_files(self):
#         directory = path.dirname(path.realpath(__file__))
#
#         data_directory = "{}/integration/data/cli".format(directory)
#
#         result = subprocess.check_output(
#             "python {0}/../autolens pipeline test "
#             "--image={1}/image.fits "
#             "--noise={1}/noise_map_.fits "
#             "--psf={1}/psf.fits "
#             "--pixel-scale=0.05".format(directory, data_directory),
#             shell=True)
#
#         assert result is not None
