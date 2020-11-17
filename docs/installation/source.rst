.. _source:

Building From Source
====================

Building from source means that you clone (or fork) the **PyAutoLens** GitHub repository and run **PyAutoLens** from
there. Unlike ``conda`` and ``pip`` this provides a build of the source code that you can edit and change, to
contribute the development **PyAutoLens** or experiment with yourself!

A large amount of **PyAutoLens** functionality is contained in its parent projects:

**PyAutoFit** https://github.com/rhayes777/PyAutoFit

**PyAutoArray** https://github.com/Jammy2211/PyAutoArray

**PyAutoGalaxy** https://github.com/Jammy2211/PyAutoGalaxy

If you wish to build from source all code used by **PyAutoLens** you may need to build from source these 3 additional
projects. We include below instructions for building just **PyAutoLens** from source or building all projects.

Building Only PyAutoLens
------------------------

First, clone (or fork) the **PyAutoLens** GitHub repository:

.. code-block:: bash

    git clone https://github.com/Jammy2211/PyAutoLens

Next, install the **PyAutoLens** dependencies via pip:

.. code-block:: bash

   pip install -r PyAutoLens/requirements.txt

If you are using a `conda` environment, add the source repository as follows:

.. code-block:: bash

   conda-develop PyAutoLens

Alternatively, if you are using a Python environment include the **PyAutoLens** source repository in your PYTHONPATH
(noting that you must replace the text ``/path/to`` with the path to the **PyAutoLens** directory on your computer):

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoLens

Finally, check the **PyAutoLens** unit tests run and pass (you may need to install pytest via ``pip install pytest``):

.. code-block:: bash

   cd /path/to/PyAutoLens
   python3 -m pytest


Building All Projects
---------------------

First, clone (or fork) all 4 GitHub repositories:

.. code-block:: bash

    git clone https://github.com/rhayes777/PyAutoFit
    git clone https://github.com/Jammy2211/PyAutoArray
    git clone https://github.com/Jammy2211/PyAutoGalaxy
    git clone https://github.com/Jammy2211/PyAutoLens

Next, install the source build dependencies of each project via pip:

.. code-block:: bash

   pip install -r PyAutoFit/requirements_source.txt
   pip install -r PyAutoArray/requirements_source.txt
   pip install -r PyAutoGalaxy/requirements_source.txt
   pip install -r PyAutoLens/requirements_source.txt

If you are using a `conda` environment, add each source repository as follows:

.. code-block:: bash

   conda-develop PyAutoFit
   conda-develop PyAutoArray
   conda-develop PyAutoGalaxy
   conda-develop PyAutoLens

Alternatively, if you are using a Python environment include each source repository in your PYTHONPATH
(noting that you must replace the text ``/path/to`` with the path to the each directory on your computer):

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoFit
   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoArray
   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoGalaxy
   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoLens

Finally, check the unit tests run and pass for each project (you may need to install pytest via ``pip install pytest``):

.. code-block:: bash

   cd /path/to/PyAutoFit
   python3 -m pytest
   cd ../PyAutoArray
   python3 -m pytest
   cd ../PyAutoGalaxy
   python3 -m pytest
   cd ../PyAutoLens
   python3 -m pytest