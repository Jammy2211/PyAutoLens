from autofit.conf import *

"""
Search for default configuration and put output in the same folder as config.

The search is performed in this order:
1) workspace. This is assumed to be in the same directory as autolens in the Docker container
2) current working directory. This is to allow for installation and use with pip where users would expect the
   configuration in their current directory to be used.
3) autolens. This is a backup for when no configuration is found. In this case it is still assumed a workspace directory
   exists in the same directory as autolens.
"""

autolens_directory = os.path.dirname(os.path.realpath(__file__))
workspace_directory = "{}/../workspace".format(autolens_directory)
current_directory = os.getcwd()

if is_config_in(workspace_directory):
    CONFIG_PATH = "{}/config".format(workspace_directory)
    instance = Config(CONFIG_PATH, "{}/output/".format(workspace_directory))
elif is_config_in(current_directory):
    CONFIG_PATH = "{}/config".format(current_directory)
    instance = Config(CONFIG_PATH, "{}/output/".format(current_directory))
else:
    CONFIG_PATH = "{}/config".format(autolens_directory)
    instance = Config(CONFIG_PATH, "{}/../workspace/output/".format(autolens_directory))
