# Environments

CellTransformer2 uses two Python environments; see the paper's Software setup for the install commands.

- `ct2-pip-freeze.txt`: full package list of the training environment used for the tested runs (`pip freeze`).
- `rapids.yml`: the main version pins of the RAPIDS environment (cuML/cuDF 24.12) used for smoothing and
  clustering. It is a specification, not an exact lock of every package.

The package dependencies are in `requirements.txt`. `click` is pinned to 8.1.8 because
`typer` 0.15.2's `--help` output needs the Click 8.1 API.
