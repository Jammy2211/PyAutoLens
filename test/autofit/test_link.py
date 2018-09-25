import os
from autolens.autofit import link


class TestCase(object):
    def test_create_dir(self):
        assert os.path.exists(link.autolens_dir)
