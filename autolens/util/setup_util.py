import os
import subprocess
import sys
from autoconf import conf

def setup_colab():

    os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

    try:
        import google.colab
    except ImportError:
        raise RuntimeError(
            """
            You are not running in a Google Colab environment so cannot use the setup_colab() function.
            
            You should therefore have PyAutoLens installed locally in your environment already (e.g. via pip or 
            conda and can run the rest of your script normally).
            """
        )

    # Install required packages
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "autoconf", "autofit", "autoarray", "autogalaxy", "autolens",
                           "pyvis==0.3.2", "dill==0.4.0", "jaxnnls",
                           "pyprojroot==0.2.0", "nautilus-sampler==1.0.4",
                           "timeout_decorator==0.5.0", "anesthetic==2.8.14",
                           "--no-deps"])

    subprocess.run([
        "git", "clone", "https://github.com/Jammy2211/autolens_workspace"
    ], check=True)

    os.chdir("/content/autolens_workspace")

    conf.instance.push(
        new_path="/content/autolens_workspace/config",
        output_path="/content/autolens_workspace/output",
    )

    print(
        """
        ***Google Colab Setup Complete, which included:***
        
        - Installation of PyAutoLens and other required packages.
        - Cloning of the autolens_workspace GitHub repository.
        - Setting up environment variables for JAX for improved performance.
        - Setting the configuration paths to the workspace config and output folders suitable for Colab.
        """
    )