# Dependency definitions and records

`setup_environment.sh` is the supported installer. It creates two independent
environments under the storage root, without requiring a global conda install.

| Component | Demonstrated version / setting |
|---|---|
| Platform | Linux x86_64, Python 3.11 |
| Reported workstation OS | Ubuntu 22.04.5 LTS; kernel 5.15.0-190-generic (user-supplied banner) |
| Bootstrap | micromamba 2.9.0-0; executable SHA256 pinned |
| Training | PyTorch 2.5.1+cu121, Lightning 2.4.0, torchdata 0.10.1 |
| Arrays / tables | NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1 |
| AnnData / HDF5 | anndata 0.11.4, h5py 3.12.1 |
| GPU clustering | RAPIDS cuML/cuDF 24.12, CUDA environment 12.2 |
| RAPIDS compatibility | numba 0.60.0, numba-cuda 0.0.17.1, pynvjitlink 0.4.0 |
| Deterministic CuBLAS | `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup |

`../requirements.txt` lists the direct pinned Python requirements.
`ct2-pip-freeze.txt` is the **actual observed full pip package list** from the
successful GPU run and supplies constraints during installation. PyTorch's CUDA
wheel is installed first from its official cu121 index. The CPU-only software
tests use the direct requirements with a CPU PyTorch wheel instead.

`rapids.yml` describes the conda channels, main compatibility pins and plotting
dependencies used by the installer. It is not an exact build-level lock.
`rapids-pip-freeze-record.txt` is an original observation from the completed run;
its `file:///...` entries are conda build paths and **must not be passed to pip as
an installation requirements file**. The historical report did not capture an
explicit conda build export, so exact original transitive conda builds cannot be
reconstructed from that record alone.

New runs now save `ct2_conda_explicit.txt`, `rapids_conda_explicit.txt` and
`rapids_conda_packages.json`, in addition to both pip freezes and hardware records.
These are included in the generated report ZIP for stronger future reproduction.
New runs also capture `/etc/os-release` as `os_release.txt` alongside `uname.txt`;
the reported OS/kernel above describes the workstation rather than setting an
exact-version requirement.

Fresh environment solves require the original package versions to remain
available on their public channels. No dependency cache or binary is stored in
Git. The new repository packaging has not been independently installed on a
second Linux GPU machine; successful run evidence and local CPU tests are
described separately in `../docs/VALIDATION.md`.
