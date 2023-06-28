# Contributing

## Setting Up Development Environment

First, [install pyenv](https://github.com/pyenv/pyenv#installation) so you can run the code under all of the supported environments. Also make sure to [install pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installation) so you can create python environments with the correct versions.

Next, run `sh pyenv_setup.sh` to download all supported python versions. This will also set the local python versions for the project properly. This will make sure that `tox run` can find all of the python versions for testing.

To create a virtual environment to run and test code locally during development, you can run the following from the base directory:

```sh
pyenv virtualenv {python_version} env-name
pyenv activate env-name
pip install -r requirements.txt
```

Simply replace `{python_version}` with the version you want the environment to use (e.g. 3.10.9) and name the environment accordingly (e.g. sotai-3.10).

## Development Workflow

1. Search through existing Issues to see if what you want to work on has already been added.
   - If not, please create a new issue. This will help to reduce duplicated work.
2. For first-time contributors, visit [https://github.com/SOTAI-Labs/sotai](https://github.com/SOTAI-Labs/sotai) and "Fork" the repository (see the button in the top right corner).

   - You'll need to set up [SSH authentication](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
   - Clone the forked project and point it to the main project:

   ```shell
   git clone https://github.com/<your-username>/sotai.git
   git remote add upstream https://github.com/SOTAI-Labs/sotai.git
   ```

3. Development.

   - Make sure you are in sync with the main repo:

   ```shell
   git checkout dev
   git pull upstream dev
   ```

   - Create a `git` feature branch with a meaningful name where you will add your contributions.

   ```shell
   git checkout -b meaningful-branch-name
   ```

   - Start coding! commit your changes locally as you work:

   ```shell
   git add sotai/modified_file.py tests/test_modified_file.py
   git commit -m "feat: specific description of changes contained in commit"
   ```

   - Format your code!

   ```shell
   python -m black sotai tests
   ```

   - Lint and test your code! From the base directory, run:

   ```shell
   tox run
   ```

4. Contributions are submitted through [GitHub Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

   - When you are ready to submit your contribution for review, push your branch:

   ```shell
   git push origin meaningful-branch-name
   ```

   - Open the printed URL to open a PR. Make sure to fill in a detailed title and description. Submit your PR for review.
   - Link the issue you selected or created under "Development"
   - We will review your contribution and add any comments to the PR. Commit any updates you make in response to comments and push them to the branch (they will be automatically included in the PR)

### Pull Requests

Please conform to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all PR titles and commits.

## Workspace Structure

```bash
sotai/
├── ...
├── docs/  # All of the documentation here is synced with the online docs.
│   ├── examples/  # Examples of how to use the library.
│   ├── notebooks/  # Jupyter Notebook examples.
│   └── ...
├── sotai/  # SOTAI SDK.
│   ├── layers/  # calibrated modeling layers
│   └──
│   ├── pipelines/  # Pipelines for calibrated.
│   ├── enums.py  # SDK enums.
│   └── ...
├── tests/  # The entire system should be unit tested. Files should have corresponding test_*.py files here.
│   └── ...
└── requirements.txt  # These packages must be installed for the current development branch to work.
```

## Formatting & Linting

In an effort to keep the codebase clean and easy to work with, we use `black` formatting and `pylint` for linting. Before sending any PR for review, make sure to run both `black` and `pylint`.

If you are using VS Code, then install the extensions in `.vscode/extensions.json` and the workspace settings should automatically run `black` formatting on save and show `pylint` errors.
