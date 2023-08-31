"""SOTAI SDK."""

# This version must always be one version ahead of the current release, so it
# matches the current state of development, which will always be ahead of the
# current release. Use Semantic Versioning.
__version__ = "0.5.0"

from . import api, constants, data, demo, features, layers, models, training, external
from .enums import *
from .pipeline import *
from .trained_model import *
from .types import *
