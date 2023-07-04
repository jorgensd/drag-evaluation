"""Top-level package for Drag evaulation for fluid flow."""
from importlib.metadata import metadata
from drag_evaluation import mesh_generation

meta = metadata("drag_evaluation")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = ["mesh_generation"]
