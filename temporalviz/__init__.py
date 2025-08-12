from . import __meta__

__version__ = __meta__.version

# Public API
from .base import create_visualizer  # noqa: F401
from .visualize_dynamics import visualize_dynamics  # noqa: F401
