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

    def test_make_linked_file(self):
        temp_file_path = "/tmp/linked_file"
        path = link.make_linked_file(temp_file_path)
        assert os.path.exists(path)
        assert os.path.exists(temp_file_path)
