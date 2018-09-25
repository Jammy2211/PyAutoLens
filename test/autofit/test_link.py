import os
from autolens.autofit import link


class TestCase(object):
    def test_create_dir(self):
        assert os.path.exists(link.autolens_dir)

    def test_consistent_dir(self):
        directory = link.path_for("/a/random/directory")
        assert link.autolens_dir in directory
        assert len(directory.split("/")[-1]) == link.SUB_PATH_LENGTH
        assert directory == link.path_for("/a/random/directory")
        assert directory != link.path_for("/b/random/directory")
