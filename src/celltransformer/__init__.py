"""celltransformer — spatial cell expression prediction package."""

from importlib.metadata import PackageNotFoundError, version

#version comes from pyproject.toml via the installed metadata
try:
    __version__ = version("celltransformer-test")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
