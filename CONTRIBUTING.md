# Contributing

Contributions are welcome and greatly appreciated!

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Jammy2211/PyAutoLens/issues

If you are playing with the PyAutoLens library and find a bug, please
reporting it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### Propose New `NonLinearSearch` or Features

The best way to send feedback is to open an issue at
https://github.com/Jammy2211/PyAutoLens
with tag *enhancement*.

If you are proposing a new `NonLinearSearch` or a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Implement `NonLinearSearch` or Features
Look through the Git issues for operator or feature requests.
Anything tagged with *enhancement* is open to whoever wants to
implement it.

### Add Examples or improve Documentation
Writing new features is not the only way to get involved and
contribute. Create examples with existing non-linear searches as well 
as improving the documentation of existing operators is as important
as making new non-linear searches and very much encouraged.


## Getting Started to contribute

Ready to contribute?

1. Follow the installation instructions for installing **PyAutoLens** from source root on our 
[readthedocs](https://pyautolens.readthedocs.io/en/latest/general/installation.html#forking-cloning>).

2. Create a branch for local development:
    ```
    git checkout -b name-of-your-branch
    ```
    Now you can make your changes locally.

3. When you're done making changes, check that old and new tests pass
succesfully:
    ```
    cd PyAutoLens/test_autolens
    python3 -m pytest
    ```

4. Commit your changes and push your branch to GitLab::
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.

5. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
