Developer Documentation
#######################

This document provides information for developers who want to contribute to the development of ``ucc``, as well as for maintainers.

Creating a release
==================

.. important::
    This section is intended for maintainers of the :code:`ucc` repository.

To release a new version of ``ucc`` on GitHub, follow the steps below.

1. **Bump the Version:**
    - Increment the version in ``pyproject.toml`` according to `semantic versioning <https://semver.org/>`_.

2. **Update the Changelog:**
    - Update the ``CHANGELOG.md`` file with all new changes, improvements, and bug fixes since the previous release.
    - You can generate an initial set of release notes using GitHub's `draft a release <https://github.com/unitaryfoundation/ucc/releases/new>`_.
      Note that you are doing this before the actual release tag, since you want the ``CHANGELOG.md`` contents to be updated as of the release tag.
      So you can add a new tag in the edit box, but do NOT publish the release yet (which would create the tag and the release.)
      Those should wait until step 5 below.

3. **Commit Changes:**
    - Commit the changes to ``pyproject.toml`` and `CHANGELOG.md` and open a PR to get the changes reviewed.

4. **Create a New Tag:**
    - Once the PR is merged, pull the changes to your local repository.
    - Create a new Git tag for the release. The tag name should match the version number (e.g., ``v1.1.0``).

    .. code-block:: bash

        git tag v1.1.0
        git push origin v1.1.0

5. **Draft a New Release on GitHub:**
    - Navigate to https://github.com/unitaryfoundation/ucc/releases to create a new release.
    - Select the newly created tag.
    - Fill in the release title and description, and copy the changelog entry for the description.
    - Publish the release.

.. tip::
    Ensure that all changes pass the tests, and the documentation builds correctly before creating a release.


Publishing a new version of UCC to PyPI (maintainers only)
==========================================================
1. Follow the steps above for creating a new release.
2. The deployment to TestPyPI should trigger automatically (only maintainers will have access).
3. Run a test of the TestPyPI deployment on your local machine:
    a. Create a new python environment ≥ our latest required version, e.g. ``python3.13 -m venv ~/.venvs/test_ucc``
    b. | Run ``pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ucc``
       | to install from the TestPiPY deployment
    c. | Run ``python -c "import ucc; print(ucc.__version__)"``.
       | This should run successfully and show the latest version of UCC.
4. If all went well in the TestPyPI step, you (as a maintainer) can go to the GH Actions and approve the deployment to real PyPI.
   If for some reason this does not work, or fails to trigger on a release, you can also manually trigger the workflow in the Github Actions tab:

   .. image:: ./img/pypi_workflow.png
      :height: 300
      :width: 600
      :alt: screenshot of pypi publishing workflow on GitHub
