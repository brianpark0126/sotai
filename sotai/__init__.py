"""SOTAI SDK"""

# This version must always be one version ahead of the current release, so it
# matches the current state of development, which will always be ahead of the
# current release. Use Semantic Versioning.
#
# NOTE: as part of the release flow, update this version immediately after release.
__version__ = "0.1.0"

from . import layers
from .enums import *
from .pipeline import *
from .trained_model import *
from .types import *
from .api import *
