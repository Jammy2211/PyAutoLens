import os


def check_workspace_path():
    if "WORKSPACE" not in os.environ:
        raise OSError(
            "\n\n#############################################\n"
            "### WORKSPACE ENVIRONMENT VARIABLE ERRROR ###\n"
            "#############################################\n\n"
            "PyAutoLens is Unable to locate the WORKSPACE environment variable. This variable must be set for "
            "PyAutoLens to know the location of the configuration files and results output folder.\n\n"
            ""
            "To set the WORKSPACE environment variable, run the 'setup_environment.py' script in the "
            "autolens_workspace. If you have not yet downloaded the autolens_workspace do this using the link below, for "
            "example if you cloned / forked the PyAutoLens repository. The GitHub repo for the autolens_workspace is:\n\n"
            ""
            "https://github.com/Jammy2211/autolens_workspace\n\n"
            ""
            "For more information on the WORKSPACE environment variable and instructions for settings it manually, see "
            "the link below:\n\n"
            ""
            "https://pyautolens.readthedocs.io/en/latest/general/installation.html#environment-variables\n\n"
        )

    from autoconf import conf

    try:

        conf.get_matplotlib_backend()
        return

    except Exception:

        raise OSError(
            "\n\n#############################################\n"
            "### WORKSPACE ENVIRONMENT VARIABLE ERRROR ###\n"
            "#############################################\n\n"
            ""
            "PyAutoLens located the following WORKSPACE environment variable:\n\n"
            ""
            f"{os.environ['WORKSPACE']}\n\n"
            ""
            "This path does not point to an autolens_workspace on your computer. Please double check the path and "
            "update the WORKSPACE path to point to a valid autolens_workspace.\n\n"
            "To reset the WORKSPACE environment variable, run the 'setup_environment.py' script in the "
            "autolens_workspace. If you have not yet downloaded the autolens_workspace do this using the link below, "
            "for example if you cloned / forked the PyAutoLens repository. The GitHub repo for the autolens_workspace is:\n\n"
            ""
            "https://github.com/Jammy2211/autolens_workspace\n\n"
            ""
            "For more information on the WORKSPACE environment variable and instructions for settings it manually, see "
            "the link below:\n\n"
            ""
            "https://pyautolens.readthedocs.io/en/latest/general/installation.html#environment-variables\n\n"
        )
