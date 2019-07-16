import os
import shutil

dirpath = os.path.dirname(os.path.realpath(__file__))


def reset_paths(test_name, output_path):

    try:
        shutil.rmtree(output_path + "/" + test_name)
    except FileNotFoundError:
        pass
